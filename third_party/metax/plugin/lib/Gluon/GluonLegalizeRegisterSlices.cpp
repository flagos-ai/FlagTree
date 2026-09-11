#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Analysis/GluonLayoutPropagation.h"
#include "Gluon/Passes.h"
#include "Gluon/Targets/GluonC500Layout.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "metax-gluon-legalize-register-slices"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;
namespace gluon_layout = ::mlir::triton::gpu::metax::gluon;
namespace c500 = ::mlir::triton::gpu::metax::gluon::c500;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONLEGALIZEREGISTERSLICES
#include "Gluon/Passes.h.inc"

namespace {

template <typename T>
static std::string formatIntegerArray(ArrayRef<T> values) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";
  return storage;
}

using gluon_layout::hasAutoEncoding;

static bool canFoldAsTensorLayoutConversion(Type sourceType,
                                            Type resultType) {
  auto sourceTensor = dyn_cast<RankedTensorType>(sourceType);
  auto resultTensor = dyn_cast<RankedTensorType>(resultType);
  if (!sourceTensor || !resultTensor ||
      sourceTensor.getShape() != resultTensor.getShape() ||
      sourceTensor.getElementType() != resultTensor.getElementType())
    return false;

  Attribute sourceEncoding = sourceTensor.getEncoding();
  Attribute resultEncoding = resultTensor.getEncoding();
  return gluon_layout::isSupportedTensorConstraintEncoding(sourceEncoding) &&
         gluon_layout::isSupportedTensorConstraintEncoding(resultEncoding) &&
         !gluon_layout::containsNoVerifyEncoding(sourceEncoding) &&
         !gluon_layout::containsNoVerifyEncoding(resultEncoding);
}

class FoldLocalAllocLoad : public OpRewritePattern<ttg::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value src = allocOp.getSrc();
    if (!src || !isa<RankedTensorType>(src.getType()))
      return failure();

    SmallVector<ttg::LocalLoadOp> loads;
    for (Operation *user : allocOp->getUsers()) {
      auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(user);
      if (!localLoadOp || localLoadOp.getToken())
        return failure();
      if (!canFoldAsTensorLayoutConversion(src.getType(),
                                           localLoadOp.getType()))
        return failure();
      loads.push_back(localLoadOp);
    }
    if (loads.empty())
      return failure();

    for (ttg::LocalLoadOp localLoadOp : loads) {
      rewriter.setInsertionPoint(localLoadOp);
      Value replacement = src;
      if (src.getType() != localLoadOp.getType()) {
        replacement = rewriter.create<ttg::ConvertLayoutOp>(
            localLoadOp.getLoc(), localLoadOp.getType(), src);
        LDBG("[local-alloc-load-fold] materialized convert_layout from "
             << src.getType() << " to " << localLoadOp.getType());
      } else {
        LDBG("[local-alloc-load-fold] folded identity staging round-trip");
      }
      rewriter.replaceOp(localLoadOp, replacement);
    }
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);
    return success();
  }
};

static LogicalResult foldLocalAllocLoads(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  patterns.add<FoldLocalAllocLoad>(module.getContext());
  return applyPatternsGreedily(module, std::move(patterns));
}

struct RegisterSliceMaterialization {
  Value fullValue;
  RankedTensorType fullType;
  c500::RegisterSliceMaterializationPlan targetPlan;
};

static FailureOr<RegisterSliceMaterialization>
prepareRegisterSliceMaterialization(Value fullValue,
                                    RankedTensorType subType,
                                    ArrayRef<int64_t> offsets,
                                    Operation *anchor,
                                    OpBuilder &builder) {
  auto fullType = dyn_cast<RankedTensorType>(fullValue.getType());
  if (!fullType)
    return failure();

  FailureOr<c500::RegisterSliceMaterializationPlan> targetPlan =
      c500::planC500RegisterSliceMaterialization(fullType, subType, offsets,
                                                 anchor);
  if (failed(targetPlan))
    return failure();
  if (!targetPlan->physicalFullType || !targetPlan->physicalSubType ||
      targetPlan->physicalFullType.getContext() != anchor->getContext() ||
      targetPlan->physicalSubType.getContext() != anchor->getContext() ||
      gluon_layout::containsAutoEncoding(targetPlan->physicalFullType) ||
      gluon_layout::containsNoVerifyEncoding(targetPlan->physicalFullType) ||
      gluon_layout::containsAutoEncoding(targetPlan->physicalSubType) ||
      gluon_layout::containsNoVerifyEncoding(targetPlan->physicalSubType) ||
      targetPlan->physicalFullType.getShape() != fullType.getShape() ||
      targetPlan->physicalFullType.getElementType() !=
          fullType.getElementType() ||
      targetPlan->physicalSubType.getShape() != subType.getShape() ||
      targetPlan->physicalSubType.getElementType() != subType.getElementType()) {
    anchor->emitError()
        << "C500 returned a malformed register-slice materialization plan";
    return failure();
  }

  RankedTensorType materializedFullType = targetPlan->physicalFullType;
  Value materializedFullValue = fullValue;
  if (materializedFullType != fullType) {
    builder.setInsertionPoint(anchor);
    materializedFullValue = ttg::ConvertLayoutOp::create(
        builder, anchor->getLoc(), materializedFullType, fullValue);
    LDBG("[register-slice-boundary] target=c500, materialize full-tile "
            "conversion from "
         << fullType.getEncoding() << " to "
         << materializedFullType.getEncoding() << " for offsets="
         << formatIntegerArray(offsets) << " and sub-shape="
         << formatIntegerArray(subType.getShape()));
  }
  LDBG("[register-slice-physical-type] target=c500, requested=" << subType
       << ", planned=" << targetPlan->physicalSubType);
  return RegisterSliceMaterialization{materializedFullValue,
                                      materializedFullType,
                                      std::move(*targetPlan)};
}

static LogicalResult rewriteGluonTensorGlueOps(tt::FuncOp func) {
  SmallVector<gluon_dialect::ExtractSliceOp> extractOps;
  SmallVector<gluon_dialect::InsertSliceOp> insertOps;
  func.walk([&](gluon_dialect::ExtractSliceOp op) { extractOps.push_back(op); });
  func.walk([&](gluon_dialect::InsertSliceOp op) { insertOps.push_back(op); });

  for (auto op : extractOps) {
    if (hasAutoEncoding(op.getSource()) || hasAutoEncoding(op.getResult()))
      return op.emitError()
             << "register slice legalization requires concrete layouts; run "
                "Gluon layout propagation before legalization";
    auto sourceTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!sourceTy || !resultTy)
      return op.emitError("register slice operands must be ranked tensors");
    OpBuilder builder(op);
    auto materialization = prepareRegisterSliceMaterialization(
        op.getSource(), resultTy, op.getOffsets(), op, builder);
    if (failed(materialization))
      return failure();
    auto replacement = builder.create<ttg::ExtractTensorOp>(
        op.getLoc(), materialization->targetPlan.physicalSubType,
        materialization->fullValue,
        builder.getDenseI64ArrayAttr(
            materialization->targetPlan.physicalSlice.ctaIndices),
        builder.getDenseI64ArrayAttr(
            materialization->targetPlan.physicalSlice.elementIndices));
    Value result = replacement.getResult();
    if (result.getType() != op.getResult().getType()) {
      result = ttg::ConvertLayoutOp::create(
          builder, op.getLoc(), op.getResult().getType(), result);
      LDBG("[register-slice-boundary] materialize physical-subview conversion "
           "from "
           << replacement.getType() << " to " << op.getResult().getType());
    }
    op.getResult().replaceAllUsesWith(result);
    op.erase();
  }

  for (auto op : insertOps) {
    if (hasAutoEncoding(op.getBase()) || hasAutoEncoding(op.getUpdate()) ||
        hasAutoEncoding(op.getResult()))
      return op.emitError()
             << "register slice_update legalization requires concrete "
                "layouts; run Gluon layout propagation before legalization";
    auto baseTy = dyn_cast<RankedTensorType>(op.getBase().getType());
    auto updateTy = dyn_cast<RankedTensorType>(op.getUpdate().getType());
    if (!baseTy || !updateTy)
      return op.emitError(
          "register slice_update operands must be ranked tensors");
    OpBuilder builder(op);
    auto materialization = prepareRegisterSliceMaterialization(
        op.getBase(), updateTy, op.getOffsets(), op, builder);
    if (failed(materialization))
      return failure();
    Value physicalUpdate = op.getUpdate();
    if (physicalUpdate.getType() !=
        materialization->targetPlan.physicalSubType)
      physicalUpdate = ttg::ConvertLayoutOp::create(
          builder, op.getLoc(), materialization->targetPlan.physicalSubType,
          physicalUpdate);
    Type insertedType = materialization->fullType;
    auto replacement = builder.create<ttg::InsertTensorOp>(
        op.getLoc(), insertedType, materialization->fullValue, physicalUpdate,
        builder.getDenseI64ArrayAttr(
            materialization->targetPlan.physicalSlice.ctaIndices),
        builder.getDenseI64ArrayAttr(
            materialization->targetPlan.physicalSlice.elementIndices));
    Value result = replacement.getResult();
    if (insertedType != op.getResult().getType())
      result = ttg::ConvertLayoutOp::create(
          builder, op.getLoc(), op.getResult().getType(), result);
    op.getResult().replaceAllUsesWith(result);
    op.erase();
  }
  return success();
}

static LogicalResult legalizeRegisterSlicesImpl(ModuleOp module) {
  if (failed(gluon_layout::verifyNoResidualCalls(
          module, "MetaX Gluon register-slice legalization")))
    return failure();
  if (failed(gluon_layout::verifyNoResidualNoVerifyEncodings(
          module, "MetaX Gluon register-slice legalization")) ||
      failed(gluon_layout::verifyNoResidualAutoEncodings(
          module, "MetaX Gluon register-slice legalization")))
    return module.emitError()
           << "Gluon register-slice legalization requires concrete layouts; "
              "run layout propagation or placeholder resolution first";

  for (tt::FuncOp func : module.getOps<tt::FuncOp>())
    if (failed(rewriteGluonTensorGlueOps(func)))
      return failure();

  // Slice lowering exposes identity local_alloc/local_load round trips. Fold
  // them only after this explicit legalization stage.
  return foldLocalAllocLoads(module);
}

class TritonMETAXGPUGluonLegalizeRegisterSlicesPass
    : public impl::TritonMETAXGPUGluonLegalizeRegisterSlicesBase<
          TritonMETAXGPUGluonLegalizeRegisterSlicesPass> {
public:
  void runOnOperation() override {
    if (failed(legalizeRegisterSlicesImpl(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
createTritonMETAXGPUGluonLegalizeRegisterSlicesPass() {
  return std::make_unique<TritonMETAXGPUGluonLegalizeRegisterSlicesPass>();
}

} // namespace mlir
