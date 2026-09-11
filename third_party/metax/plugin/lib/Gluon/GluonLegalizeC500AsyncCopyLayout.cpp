#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Analysis/GluonDotAnalysis.h"
#include "Gluon/GluonC500AsyncCopyLayout.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/Passes.h"
#include "Gluon/GluonC500AsyncCopyPlan.h"
#include "Gluon/Targets/GluonC500Layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "metax-gluon-legalize-c500-async-copy-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace c500 = ::mlir::triton::gpu::metax::gluon::c500;
namespace gluon_layout = ::mlir::triton::gpu::metax::gluon;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONLEGALIZEC500ASYNCCOPYLAYOUT
#include "Gluon/Passes.h.inc"

namespace {

static RankedTensorType cloneTensorTypeWithEncoding(RankedTensorType type,
                                                    Attribute encoding) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encoding);
}

static Value convertTensorEncoding(Value value, Attribute encoding,
                                   OpBuilder &builder, Location loc) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getEncoding() == encoding)
    return value;
  auto targetType = cloneTensorTypeWithEncoding(type, encoding);
  return builder.create<ttg::ConvertLayoutOp>(loc, targetType, value)
      .getResult();
}

static Value stripLayoutCasts(Value value) {
  while (value) {
    if (auto require = value.getDefiningOp<gluon_dialect::RequireLayoutOp>()) {
      value = require.getSrc();
      continue;
    }
    if (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = convert.getSrc();
      continue;
    }
    return value;
  }
  return {};
}

static bool hasMatchingSwizzleTensor(Value value, unsigned inVec,
                                     ttg::SwizzledSharedEncodingAttr shared) {
  auto swizzle = value.getDefiningOp<ttg::SwizzleTensorOp>();
  return swizzle && swizzle.getInVec() == inVec &&
         swizzle.getOutVec() == shared.getVec() &&
         swizzle.getPerPhase() == shared.getPerPhase() &&
         swizzle.getMaxPhase() == shared.getMaxPhase();
}

static bool canMaterializeTensorAtShape(Value value,
                                        ArrayRef<int64_t> resultShape) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != static_cast<int64_t>(resultShape.size()))
    return false;
  return llvm::all_of(llvm::zip(type.getShape(), resultShape), [](auto extent) {
    auto [sourceExtent, resultExtent] = extent;
    return sourceExtent == resultExtent || sourceExtent == 1;
  });
}

static FailureOr<Value> materializeTensorAtShape(
    Value value, ArrayRef<int64_t> resultShape, Attribute targetEncoding,
    OpBuilder &builder, Location loc) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != static_cast<int64_t>(resultShape.size()))
    return failure();

  Value converted = convertTensorEncoding(value, targetEncoding, builder, loc);
  if (type.getShape() == resultShape)
    return converted;
  for (auto [sourceExtent, resultExtent] :
       llvm::zip(type.getShape(), resultShape))
    if (sourceExtent != resultExtent && sourceExtent != 1)
      return failure();

  auto resultType = RankedTensorType::get(
      resultShape, type.getElementType(), targetEncoding);
  return builder.create<tt::BroadcastOp>(loc, resultType, converted).getResult();
}

enum class CmpOperandRole { Constant, Contiguous, Unsupported };

/// A side-effect-free recipe for rebuilding one mask value at the async-copy
/// source shape. Recipe nodes only retain existing SSA values and operations;
/// no IR is created until every async-copy and BSM contract in the module has
/// been analyzed successfully.
struct MaskRewriteRecipe {
  enum class Kind { TensorAtShape, Compare, Elementwise };

  Kind kind;
  Value source;
  CmpOperandRole lhsRole = CmpOperandRole::Unsupported;
  CmpOperandRole rhsRole = CmpOperandRole::Unsupported;
  SmallVector<unsigned> tensorOperandRecipes;
};

struct MaskRewritePlan {
  SmallVector<MaskRewriteRecipe> recipes;
  SmallVector<int64_t> resultShape;
  unsigned rootRecipe = 0;
};

static FailureOr<CmpOperandRole> classifyCmpOperand(
    Value operand, ttg::SwizzledSharedEncodingAttr shared,
    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  unsigned contiguousDim = shared.getOrder().front();
  if (!type || contiguousDim >= static_cast<unsigned>(type.getRank()) ||
      ShapedType::isDynamic(type.getShape()[contiguousDim]))
    return failure();

  tt::AxisInfo *axisInfo = axisInfoAnalysis.getAxisInfo(operand);
  if (!axisInfo)
    return failure();
  int64_t extent = type.getShape()[contiguousDim];
  if (axisInfo->getConstancy(contiguousDim) >= extent)
    return CmpOperandRole::Constant;
  if (axisInfo->getContiguity(contiguousDim) >= extent)
    return CmpOperandRole::Contiguous;
  return CmpOperandRole::Unsupported;
}

static arith::CmpIPredicate
swapCmpPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return predicate;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unknown arith.cmpi predicate");
}

static bool isVectorUniformCmpMask(
    arith::CmpIOp cmp, CmpOperandRole lhsRole, CmpOperandRole rhsRole,
    unsigned sourceVec, ttg::SwizzledSharedEncodingAttr shared,
    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  if (sourceVec <= 1 || (lhsRole != CmpOperandRole::Contiguous &&
                         rhsRole != CmpOperandRole::Contiguous))
    return true;

  bool coordinateIsLhs = lhsRole == CmpOperandRole::Contiguous;
  Value coordinate = coordinateIsLhs ? cmp.getLhs() : cmp.getRhs();
  Value boundary = coordinateIsLhs ? cmp.getRhs() : cmp.getLhs();
  arith::CmpIPredicate predicate =
      coordinateIsLhs ? cmp.getPredicate()
                      : swapCmpPredicate(cmp.getPredicate());
  tt::AxisInfo *coordinateInfo = axisInfoAnalysis.getAxisInfo(coordinate);
  tt::AxisInfo *boundaryInfo = axisInfoAnalysis.getAxisInfo(boundary);
  unsigned contiguousDim = shared.getOrder().front();
  if (!coordinateInfo || !boundaryInfo ||
      coordinateInfo->getDivisibility(contiguousDim) < sourceVec)
    return false;

  switch (predicate) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return boundaryInfo->getDivisibility(contiguousDim) >= sourceVec;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt: {
    std::optional<int64_t> constant = boundaryInfo->getConstantValue();
    return constant &&
           (static_cast<uint64_t>(*constant) + 1) % sourceVec == 0;
  }
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return false;
  }
  llvm_unreachable("unknown arith.cmpi predicate");
}

static bool canMaterializeSwizzledCmpOperand(
    Value operand, CmpOperandRole role, ArrayRef<int64_t> resultShape,
    Attribute targetEncoding, unsigned sourceVec,
    ttg::SwizzledSharedEncodingAttr shared) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  if (!type || !canMaterializeTensorAtShape(operand, resultShape))
    return false;
  if (role == CmpOperandRole::Constant)
    return true;
  if (role != CmpOperandRole::Contiguous)
    return false;

  bool preservesValue = type.getShape() == resultShape &&
                        type.getEncoding() == targetEncoding;
  if (preservesValue && hasMatchingSwizzleTensor(operand, sourceVec, shared))
    return true;
  return type.getElementType().isIntOrIndex();
}

class MaskRewritePlanner {
public:
  MaskRewritePlanner(Attribute targetEncoding, unsigned sourceVec,
                     ttg::SwizzledSharedEncodingAttr shared,
                     tt::ModuleAxisInfoAnalysis &axisInfoAnalysis)
      : targetEncoding(targetEncoding), sourceVec(sourceVec), shared(shared),
        axisInfoAnalysis(axisInfoAnalysis) {}

  FailureOr<MaskRewritePlan> analyze(Value mask) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskType)
      return failure();
    plan.resultShape.assign(maskType.getShape().begin(),
                            maskType.getShape().end());
    FailureOr<unsigned> root = analyzeImpl(mask);
    if (failed(root))
      return failure();
    plan.rootRecipe = *root;
    return std::move(plan);
  }

private:
  FailureOr<unsigned> analyzeImpl(Value mask) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskType)
      return failure();
    bool preservesShape =
        maskType.getShape() == ArrayRef<int64_t>(plan.resultShape);
    if (preservesShape) {
      auto cached = shapePreservingRecipes.find(mask);
      if (cached != shapePreservingRecipes.end())
        return cached->second;
    }
    if (!active.insert(mask).second)
      return failure();
    auto activeGuard = llvm::make_scope_exit([&] { active.erase(mask); });

    auto record = [&](MaskRewriteRecipe recipe) {
      unsigned recipeId = plan.recipes.size();
      plan.recipes.push_back(std::move(recipe));
      if (preservesShape)
        shapePreservingRecipes.try_emplace(mask, recipeId);
      return recipeId;
    };

    if (c500::isUniformTensorValue(mask)) {
      if (!canMaterializeTensorAtShape(mask, plan.resultShape))
        return failure();
      return record({MaskRewriteRecipe::Kind::TensorAtShape, mask});
    }

    Value stripped = stripLayoutCasts(mask);
    if (stripped != mask) {
      FailureOr<unsigned> recipe = analyzeImpl(stripped);
      if (succeeded(recipe) && preservesShape)
        shapePreservingRecipes.try_emplace(mask, *recipe);
      return recipe;
    }

    if (auto broadcast = mask.getDefiningOp<tt::BroadcastOp>()) {
      FailureOr<unsigned> recipe = analyzeImpl(broadcast.getSrc());
      if (succeeded(recipe) && preservesShape)
        shapePreservingRecipes.try_emplace(mask, *recipe);
      return recipe;
    }

    auto cmp = stripped ? stripped.getDefiningOp<arith::CmpIOp>() : nullptr;
    if (cmp) {
      FailureOr<CmpOperandRole> lhsRole =
          classifyCmpOperand(cmp.getLhs(), shared, axisInfoAnalysis);
      FailureOr<CmpOperandRole> rhsRole =
          classifyCmpOperand(cmp.getRhs(), shared, axisInfoAnalysis);
      if (failed(lhsRole) || failed(rhsRole) ||
          *lhsRole == CmpOperandRole::Unsupported ||
          *rhsRole == CmpOperandRole::Unsupported)
        return failure();
      unsigned contiguousOperands =
          (*lhsRole == CmpOperandRole::Contiguous) +
          (*rhsRole == CmpOperandRole::Contiguous);
      if (contiguousOperands > 1)
        return failure();
      if (!isVectorUniformCmpMask(cmp, *lhsRole, *rhsRole, sourceVec, shared,
                                  axisInfoAnalysis)) {
        LDBG("[async-copy-mask] rejected vector-nonuniform cmp predicate="
             << cmp.getPredicate() << " source-vec=" << sourceVec);
        return failure();
      }
      if (!canMaterializeSwizzledCmpOperand(
              cmp.getLhs(), *lhsRole, plan.resultShape, targetEncoding,
              sourceVec, shared) ||
          !canMaterializeSwizzledCmpOperand(
              cmp.getRhs(), *rhsRole, plan.resultShape, targetEncoding,
              sourceVec, shared))
        return failure();

      MaskRewriteRecipe recipe{MaskRewriteRecipe::Kind::Compare, mask};
      recipe.lhsRole = *lhsRole;
      recipe.rhsRole = *rhsRole;
      return record(std::move(recipe));
    }

    Operation *def = mask.getDefiningOp();
    auto resultType = dyn_cast<RankedTensorType>(mask.getType());
    if (!def || !resultType || !resultType.getElementType().isInteger(1) ||
        def->getNumResults() != 1 || def->getNumRegions() != 0 ||
        !isMemoryEffectFree(def) ||
        !gluon_dialect::hasSameTensorEncodingRelation(def))
      return failure();

    MaskRewriteRecipe recipe{MaskRewriteRecipe::Kind::Elementwise, mask};
    for (Value operand : def->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType)
        continue;
      if (!operandType.getElementType().isInteger(1))
        return failure();
      FailureOr<unsigned> operandRecipe = analyzeImpl(operand);
      if (failed(operandRecipe))
        return failure();
      recipe.tensorOperandRecipes.push_back(*operandRecipe);
    }
    return record(std::move(recipe));
  }

  Attribute targetEncoding;
  unsigned sourceVec;
  ttg::SwizzledSharedEncodingAttr shared;
  tt::ModuleAxisInfoAnalysis &axisInfoAnalysis;
  MaskRewritePlan plan;
  DenseMap<Value, unsigned> shapePreservingRecipes;
  llvm::DenseSet<Value> active;
};

static FailureOr<Value> materializeSwizzledCmpOperand(
    Value operand, CmpOperandRole role, Attribute targetEncoding,
    unsigned sourceVec,
    ttg::SwizzledSharedEncodingAttr shared,
    OpBuilder &builder, Location loc) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  if (!type)
    return failure();

  Value converted = convertTensorEncoding(operand, targetEncoding, builder, loc);
  if (role == CmpOperandRole::Constant ||
      hasMatchingSwizzleTensor(converted, sourceVec, shared))
    return converted;
  if (role != CmpOperandRole::Contiguous)
    return failure();

  auto convertedType = cast<RankedTensorType>(converted.getType());
  Type elemType = convertedType.getElementType();
  if (!elemType.isIntOrIndex())
    return failure();

  // C500 async copy applies the shared-memory swizzle as part of the global
  // load/shared write instruction. For non-uniform predicates, rebuild the
  // integer expression in the same source-side swizzled coordinate system.
  return builder
      .create<ttg::SwizzleTensorOp>(
          loc, convertedType, converted, sourceVec, shared.getVec(),
          shared.getPerPhase(), shared.getMaxPhase())
      .getResult();
}

static FailureOr<Value> materializeMaskRewriteRecipe(
    unsigned recipeId, const MaskRewritePlan &plan, Attribute targetEncoding,
    unsigned sourceVec, ttg::SwizzledSharedEncodingAttr shared,
    OpBuilder &builder, Location loc, SmallVectorImpl<Value> &materialized) {
  if (recipeId >= plan.recipes.size() ||
      recipeId >= static_cast<unsigned>(materialized.size()))
    return failure();
  if (materialized[recipeId])
    return materialized[recipeId];

  const MaskRewriteRecipe &recipe = plan.recipes[recipeId];
  Value rewritten;
  switch (recipe.kind) {
  case MaskRewriteRecipe::Kind::TensorAtShape: {
    FailureOr<Value> value = materializeTensorAtShape(
        recipe.source, plan.resultShape, targetEncoding, builder, loc);
    if (failed(value))
      return failure();
    rewritten = *value;
    break;
  }
  case MaskRewriteRecipe::Kind::Compare: {
    auto cmp = recipe.source.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return failure();
    FailureOr<Value> expandedLhs = materializeTensorAtShape(
        cmp.getLhs(), plan.resultShape, targetEncoding, builder, loc);
    FailureOr<Value> expandedRhs = materializeTensorAtShape(
        cmp.getRhs(), plan.resultShape, targetEncoding, builder, loc);
    if (failed(expandedLhs) || failed(expandedRhs))
      return failure();
    FailureOr<Value> lhs = materializeSwizzledCmpOperand(
        *expandedLhs, recipe.lhsRole, targetEncoding, sourceVec, shared,
        builder, loc);
    FailureOr<Value> rhs = materializeSwizzledCmpOperand(
        *expandedRhs, recipe.rhsRole, targetEncoding, sourceVec, shared,
        builder, loc);
    if (failed(lhs) || failed(rhs) || (*lhs).getType() != (*rhs).getType())
      return failure();
    rewritten =
        builder.create<arith::CmpIOp>(loc, cmp.getPredicate(), *lhs, *rhs);
    break;
  }
  case MaskRewriteRecipe::Kind::Elementwise: {
    Operation *def = recipe.source.getDefiningOp();
    if (!def)
      return failure();
    IRMapping mapping;
    unsigned tensorOperand = 0;
    for (Value operand : def->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType())) {
        mapping.map(operand, operand);
        continue;
      }
      if (tensorOperand >= recipe.tensorOperandRecipes.size())
        return failure();
      FailureOr<Value> value = materializeMaskRewriteRecipe(
          recipe.tensorOperandRecipes[tensorOperand++], plan, targetEncoding,
          sourceVec, shared, builder, loc, materialized);
      if (failed(value))
        return failure();
      mapping.map(operand, *value);
    }
    if (tensorOperand != recipe.tensorOperandRecipes.size())
      return failure();
    Operation *cloned = builder.clone(*def, mapping);
    rewritten = cloned->getResult(0);
    auto sourceType = cast<RankedTensorType>(recipe.source.getType());
    rewritten.setType(RankedTensorType::get(
        plan.resultShape, sourceType.getElementType(), targetEncoding));
    break;
  }
  }
  materialized[recipeId] = rewritten;
  return rewritten;
}

static FailureOr<Value> materializeSourceSideSwizzledMask(
    const MaskRewritePlan &plan, Attribute targetEncoding, unsigned sourceVec,
    ttg::SwizzledSharedEncodingAttr shared, OpBuilder &builder, Location loc) {
  SmallVector<Value> materialized(plan.recipes.size());
  return materializeMaskRewriteRecipe(plan.rootRecipe, plan, targetEncoding,
                                      sourceVec, shared, builder, loc,
                                      materialized);
}

static Value materializeAllTrueMask(RankedTensorType sourceType,
                                    Attribute encoding, OpBuilder &builder,
                                    Location loc) {
  auto maskType = RankedTensorType::get(
      sourceType.getShape(), builder.getI1Type(), encoding);
  auto value = DenseElementsAttr::get(maskType, true);
  return builder.create<arith::ConstantOp>(loc, maskType, value);
}

static bool isZeroOtherValueImpl(Value value, llvm::DenseSet<Value> &visited) {
  value = stripLayoutCasts(value);
  if (!value || !visited.insert(value).second)
    return false;
  if (matchPattern(value, m_Zero()))
    return true;
  if (auto splat = value.getDefiningOp<tt::SplatOp>())
    return isZeroOtherValueImpl(splat.getSrc(), visited);
  if (auto trunc = value.getDefiningOp<arith::TruncFOp>())
    return isZeroOtherValueImpl(trunc.getIn(), visited);
  if (auto ext = value.getDefiningOp<arith::ExtFOp>())
    return isZeroOtherValueImpl(ext.getIn(), visited);
  if (auto swizzle = value.getDefiningOp<ttg::SwizzleTensorOp>())
    return isZeroOtherValueImpl(swizzle.getSrc(), visited);
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getValue().isZero();
    if (auto floating = dyn_cast<FloatAttr>(constant.getValue()))
      return floating.getValue().isZero();
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue())) {
      if (!dense.isSplat())
        return false;
      Attribute element = dense.getSplatValue<Attribute>();
      if (auto integer = dyn_cast<IntegerAttr>(element))
        return integer.getValue().isZero();
      if (auto floating = dyn_cast<FloatAttr>(element))
        return floating.getValue().isZero();
    }
  }
  return false;
}

static bool isZeroOtherValue(Value value) {
  llvm::DenseSet<Value> visited;
  return isZeroOtherValueImpl(value, visited);
}

struct AsyncCopyRewritePlan {
  ttg::AsyncCopyGlobalToLocalOp copyOp;
  c500::AsyncCopyIssuePlan issue;
  std::optional<MaskRewritePlan> maskRewrite;
  bool materializeAllTrueMask = false;
};

/// Analyze one copy and append its complete canonicalization plan. This
/// routine is intentionally mutation-free so a later invalid copy or BSM
/// contract cannot leave earlier copies partially legalized.
static LogicalResult analyzeAsyncCopy(
    ttg::AsyncCopyGlobalToLocalOp copyOp,
    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    SmallVectorImpl<AsyncCopyRewritePlan> &rewritePlans) {
  auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
  auto dstTy = dyn_cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  if (!srcTy || !dstTy)
    return copyOp.emitError()
           << "C500 async_copy legalization requires ranked tensor source "
              "and memdesc destination";

  auto shared =
      dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
  // The source ownership is a late issue-level choice. A dot/shared
  // requirement may leave the pointer expression in another concrete
  // distributed layout; derive the legal blocked issuer from address
  // contiguity and insert that local conversion below. The shared encoding,
  // by contrast, is the already-selected hard destination contract.
  if (!shared)
    return c500::verifyC500AsyncCopyIssuePlan(copyOp);

  FailureOr<c500::Rank2ContiguousOrderInfo> addressInfo =
      c500::inferRank2ContiguousOrder(copyOp.getSrc(), &axisInfoAnalysis,
                                     copyOp);
  if (failed(addressInfo))
    return failure();
  if (addressInfo->maxContiguousElements == 0)
    return copyOp.emitError()
           << "C500 async_copy legalization requires proven global address "
              "contiguity; register ownership encoding is insufficient";
  c500::AsyncCopyAddressContiguity addressContiguity{
      addressInfo->order.front(), addressInfo->maxContiguousElements};
  FailureOr<c500::AsyncCopyIssuePlan> plan =
      c500::selectC500AsyncCopyIssuePlan(
          copyOp, addressInfo->order, addressContiguity, shared);
  if (failed(plan))
    return copyOp.emitError()
           << "C500 async_copy has no physical issue plan within the proven "
              "global address-contiguity width "
           << addressInfo->maxContiguousElements;

  if (Value other = copyOp.getOther()) {
    if (!isZeroOtherValue(other))
      return copyOp.emitError()
             << "C500 async_copy currently requires other to be absent or "
                "provably zero";
  }

  AsyncCopyRewritePlan rewritePlan{copyOp, *plan};
  rewritePlan.materializeAllTrueMask = !copyOp.getMask();
  if (Value mask = copyOp.getMask();
      mask && plan->requiresSourceSideMaskSwizzle) {
    MaskRewritePlanner planner(plan->sourceEncoding, plan->copyElements,
                               shared, axisInfoAnalysis);
    FailureOr<MaskRewritePlan> maskPlan = planner.analyze(mask);
    if (failed(maskPlan))
      return copyOp.emitError()
             << "C500 async_copy has a non-uniform mask for a swizzled shared "
                "destination that cannot be rebuilt as a pure elementwise i1 "
                "expression with source-side swizzle_tensor";
    rewritePlan.maskRewrite = std::move(*maskPlan);
  }
  LDBG("[async-copy-legalization-plan] source="
       << plan->sourceEncoding << " contiguous-dim=" << plan->contiguousDim
       << " max-contiguous-elements=" << plan->maxContiguousElements
       << " address-provenance="
       << c500::stringifyRank2OrderProvenance(addressInfo->provenance)
       << " copy-bytes=" << plan->copyBytes
       << " add-mask=" << rewritePlan.materializeAllTrueMask
       << " mask-recipes="
       << (rewritePlan.maskRewrite ? rewritePlan.maskRewrite->recipes.size()
                                   : 0));
  rewritePlans.push_back(std::move(rewritePlan));
  return success();
}

static LogicalResult
materializeAsyncCopyRewrite(const AsyncCopyRewritePlan &plan) {
  ttg::AsyncCopyGlobalToLocalOp copyOp = plan.copyOp;
  Attribute targetEncoding = plan.issue.sourceEncoding;
  OpBuilder builder(copyOp);
  Location loc = copyOp.getLoc();
  auto sourceType = cast<RankedTensorType>(copyOp.getSrc().getType());

  copyOp->setOperand(
      0, convertTensorEncoding(copyOp.getSrc(), targetEncoding, builder, loc));

  if (plan.materializeAllTrueMask) {
    Value mask =
        materializeAllTrueMask(sourceType, targetEncoding, builder, loc);
    copyOp.getMaskMutable().assign(mask);
    LDBG("[async-copy-mask] materialized all-true mask for maskless copy");
  } else if (plan.maskRewrite) {
    FailureOr<Value> mask = materializeSourceSideSwizzledMask(
        *plan.maskRewrite, targetEncoding, plan.issue.copyElements,
        plan.issue.sharedEncoding, builder, loc);
    if (failed(mask))
      return copyOp.emitError()
             << "failed to materialize the prevalidated C500 async_copy mask "
                "rewrite plan";
    copyOp.getMaskMutable().assign(*mask);
    LDBG("[async-copy-mask] rebuilt non-uniform mask in source-side "
         "swizzled coordinates");
  } else {
    Value mask = convertTensorEncoding(copyOp.getMask(), targetEncoding,
                                       builder, loc);
    copyOp.getMaskMutable().assign(mask);
  }

  if (Value other = copyOp.getOther()) {
    Value converted =
        convertTensorEncoding(other, targetEncoding, builder, loc);
    copyOp.getOtherMutable().assign(converted);
  }

  // Selection and follower rewriteability were proven before mutation. Keep
  // this verifier as an internal consistency check on the materializer.
  return c500::verifyC500AsyncCopyIssuePlan(copyOp);
}

static LogicalResult legalizeC500AsyncCopyLayoutImpl(ModuleOp module) {
  if (failed(gluon_layout::verifyNoResidualCalls(
          module, "MetaX Gluon C500 async-copy legalization")) ||
      failed(gluon_layout::verifyNoResidualNoVerifyEncodings(
          module, "MetaX Gluon C500 async-copy legalization")) ||
      failed(gluon_layout::verifyNoResidualAutoEncodings(
          module, "MetaX Gluon C500 async-copy legalization")))
    return failure();

  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(module);
  SmallVector<AsyncCopyRewritePlan> rewritePlans;
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
      if (failed(
              analyzeAsyncCopy(copyOp, axisInfoAnalysis, rewritePlans)))
        return WalkResult::interrupt();
    if (auto bsm = dyn_cast<ttg::BsmPermOp>(op))
      if (failed(gluon_layout::verifyBsmPermPhysicalContract(bsm)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  LDBG("[async-copy-legalization-plan] materializing "
       << rewritePlans.size() << " copy rewrites after module validation");
  for (const AsyncCopyRewritePlan &plan : rewritePlans)
    if (failed(materializeAsyncCopyRewrite(plan)))
      return failure();

  // The marker distinguishes frontend placeholder shared-layout family roots
  // while requirements are being planned. Manual-layout pipelines
  // intentionally skip that planning pass, so this first concrete-layout
  // legalizer also owns the fallback cleanup boundary. No lower-level pass may
  // interpret the marker.
  unsigned removedDefaultLayoutMarkers = 0;
  module.walk([&](Operation *op) {
    if (gluon_layout::isCompilerManagedSharedFamilyRoot(op) &&
        op->removeAttr(gluon_layout::kDefaultSharedLayoutAttrName))
      ++removedDefaultLayoutMarkers;
  });
  LDBG("[frontend-marker-cleanup] removed-default-shared-layout-markers="
       << removedDefaultLayoutMarkers);
  return success();
}

class TritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass
    : public impl::TritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutBase<
          TritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass> {
public:
  void runOnOperation() override {
    if (failed(legalizeC500AsyncCopyLayoutImpl(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult legalizeTritonMETAXGPUGluonC500AsyncCopyLayout(ModuleOp module) {
  return legalizeC500AsyncCopyLayoutImpl(module);
}

std::unique_ptr<Pass>
createTritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass() {
  return std::make_unique<
      TritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass>();
}

} // namespace mlir
