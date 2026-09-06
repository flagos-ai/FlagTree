#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONCOMBINEOPS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool isZero(Value val) {
  return (matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()));
}

bool isAddPtrOffsetCombinable(Value first, Value second) {
  auto GetConstantIntValue = [](Value val) -> std::optional<llvm::APInt> {
    DenseElementsAttr constAttr;
    auto defOp = val.getDefiningOp();
    if (defOp) {
      if (auto splatOp = llvm::dyn_cast<SplatOp>(defOp))
        val = splatOp.getSrc();
      else if (matchPattern(defOp, m_Constant(&constAttr)) &&
               constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        // Check IntegerAttr
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
    }

    // Check constant value.
    llvm::APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal;

    return std::nullopt;
  };

  if (first.getType() == second.getType()) {
    // Whether bitwidth of element type is equal to pointer
    if (getElementTypeOrSelf(first.getType()).getIntOrFloatBitWidth() == 64)
      return true;

    // first + second does not overflow
    auto firstVal = GetConstantIntValue(first);
    auto secondVal = GetConstantIntValue(second);
    if (firstVal && secondVal) {
      bool overflow = false;
      auto resVal = firstVal->sadd_ov(*secondVal, overflow);
      return !overflow;
    }
  }
  return false;
}

#ifdef USE_MACA
bool isFmaCombinable(Operation *op) {
  auto op_mul = llvm::dyn_cast_or_null<mlir::arith::MulFOp>(op);
  if (!op_mul) {
    return false;
  }

  Type result_type = op_mul.getResult().getType(); // tensortype or scalar type
  if (auto ty = llvm::dyn_cast<RankedTensorType>(result_type)) {
    result_type = ty.getElementType();
  }

  if (op_mul.getResult().hasOneUse() && result_type &&
      (result_type.isF16() || result_type.isF32() || result_type.isF64())) {
    return true;
  }
  return false;
}
#endif

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
class CombineSelectMaskedLoadPattern : public RewritePattern {
public:
  CombineSelectMaskedLoadPattern(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 3, context,
                       {LoadOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<arith::SelectOp>(op);
    if (!selectOp)
      return failure();

    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    Value condSelect = selectOp.getCondition();

    auto loadOp = trueValue.getDefiningOp<LoadOp>();
    if (!loadOp)
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto splatOp = mask.getDefiningOp<SplatOp>();
    if (!splatOp)
      return failure();

    auto splatCond = splatOp.getSrc();
    if (splatCond != condSelect)
      return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), /*other=*/falseValue,
        loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    return success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public RewritePattern {
private:
  static bool isAddF32(const Operation *op) {
    if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
      return addf.getType().getIntOrFloatBitWidth() <= 32;
    return false;
  }

public:
  CombineBroadcastMulReducePattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reduceOp = llvm::dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();
    // only support reduce with simple addition
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddF32(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return failure();
    // operand of reduce has to be mul
    auto mulOp = reduceOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = mulOp.getOperand(0).getDefiningOp<BroadcastOp>();
    if (!broadcastLhsOp)
      return failure();
    auto broadcastRhsOp = mulOp.getOperand(1).getDefiningOp<BroadcastOp>();
    if (!broadcastRhsOp)
      return failure();
    // broadcast operand is expand dims
    auto expandLhsOp = broadcastLhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandLhsOp)
      return failure();
    auto expandRhsOp = broadcastRhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandRhsOp)
      return failure();
    // get not-broadcast dimensions
    int expandLhsAxis = expandLhsOp.getAxis();
    int expandRhsAxis = expandRhsOp.getAxis();
    if (expandLhsAxis != 2 || expandRhsAxis != 0)
      return failure();
    auto broadcastLhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    auto broadcastRhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
      return failure();
    Type newAccType = RankedTensorType::get(
        {broadcastLhsShape[0], broadcastRhsShape[2]},
        cast<ShapedType>(broadcastLhsOp.getSrc().getType()).getElementType());
    rewriter.setInsertionPoint(op);
    auto newAcc =
        SplatOp::create(rewriter, op->getLoc(), newAccType,
                        arith::ConstantOp::create(rewriter, op->getLoc(),
                                                  rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<DotOp>(op, expandLhsOp.getSrc(),
                                       expandRhsOp.getSrc(), newAcc,
                                       InputPrecision::TF32, 0);
    return success();
  }
};

// When reducing a 1D tensor the order of elements of the tensor doesn't matter.
// Therefore we can relax the reshape to allow it to re-order elements.
class CombineReshapeReducePatterns : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (reshapeOp.getAllowReorder())
      return failure();
    if (reshapeOp.getType().getRank() != 1)
      return failure();
    for (Operation *user : reshapeOp->getUsers()) {
      if (!isa<triton::ReduceOp, triton::HistogramOp>(user))
        return failure();
    }
    rewriter.modifyOpInPlace(reshapeOp,
                             [&]() { reshapeOp.setAllowReorder(true); });
    return success();
  }
};

class RankedReduceDescriptorLoads : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loadDef = reshapeOp.getSrc().getDefiningOp<triton::DescriptorLoadOp>();
    if (!loadDef || !loadDef->hasOneUse())
      return failure();
    int loadRank = loadDef.getType().getRank();
    int reshapeRank = reshapeOp.getType().getRank();
    if (!(reshapeRank < loadRank))
      return failure();
    ArrayRef<int64_t> loadShape = loadDef.getType().getShape();
    ArrayRef<int64_t> reshapeShape = reshapeOp.getType().getShape();
    for (int i = 0; i < loadRank - reshapeRank; ++i) {
      // Only rank reduce unit dims.
      if (loadShape[i] != 1)
        return failure();
    }
    if (loadShape.take_back(reshapeRank) != reshapeShape)
      return failure();
    rewriter.modifyOpInPlace(
        loadDef, [&]() { loadDef.getResult().setType(reshapeOp.getType()); });
    rewriter.replaceOp(reshapeOp, loadDef.getResult());
    return success();
  }
};

template <typename OpTy>
class CombineDotAddPattern : public mlir::OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(OpTy addOp, mlir::PatternRewriter &rewriter) const override {
    auto dotOp = addOp.getRhs().template getDefiningOp<DotOp>();
    bool isDotLHS = false;
    if (!dotOp) {
      dotOp = addOp.getLhs().template getDefiningOp<DotOp>();
      if (!dotOp) {
        return failure();
      }
      isDotLHS = true;
    }
    if (!dotOp->hasOneUse()) {
      return failure();
    }
    if (!isZero(dotOp.getC()))
      return failure();
    if constexpr (std::is_same_v<OpTy, arith::AddFOp>) {
      if (dotOp.getMaxNumImpreciseAcc() != 0) {
        return failure();
      }
    }
    rewriter.modifyOpInPlace(dotOp, [&] {
      dotOp.getCMutable().assign(isDotLHS ? addOp.getRhs() : addOp.getLhs());
      dotOp->moveBefore(addOp);
    });
    rewriter.replaceAllUsesWith(addOp, dotOp.getResult());
    return success();
  }
};

// AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddFOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddIOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
// AddFOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
using CombineDotAddIPattern = CombineDotAddPattern<arith::AddIOp>;
using CombineDotAddFPattern = CombineDotAddPattern<arith::AddFOp>;

#ifdef USE_MACA
// addf(mulf(a, b), c) => fma(a, b, c)
// addf(c, mulf(a, b)) => fma(a, b, c)
class CombineAddfmulfPattern : public mlir::RewritePattern {
public:
  CombineAddfmulfPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::AddFOp::getOperationName(), 3,
                             context, {mlir::math::FmaOp::getOperationName()}) {
  }

  mlir::LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto AddOp = llvm::dyn_cast<mlir::arith::AddFOp>(op);
    if (!AddOp)
      return mlir::failure();
    int idx = 0;
    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (isFmaCombinable(definingOp)) {
        auto op_mul = llvm::dyn_cast<mlir::arith::MulFOp>(definingOp);
        auto mul_inputs = op_mul->getOperands();
        auto FmaOp = rewriter.create<mlir::math::FmaOp>(
            op->getLoc(), op->getResultTypes()[0], mul_inputs[0], mul_inputs[1],
            op->getOperands()[idx ^ 1]);
        rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(
            op, FmaOp.getA(), FmaOp.getB(), FmaOp.getC());
        return mlir::success();
      }
      idx += 1;
    }
    return mlir::failure();
  }
};

// subf(mulf(a, b), c) => fma(a, b, mulf(c, splat(constant(-1.0))))
// subf(c, mulf(a, b)) => fma(a, mulf(b, splat(constant(-1.0))), c)
class CombineSubfmulfPattern : public mlir::RewritePattern {
public:
  CombineSubfmulfPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::SubFOp::getOperationName(), 3,
                             context, {mlir::math::FmaOp::getOperationName()}) {
  }

  mlir::LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto sub_op = llvm::dyn_cast<mlir::arith::SubFOp>(op);
    if (!sub_op)
      return mlir::failure();

    if (auto definingOp = op->getOperands()[0].getDefiningOp()) {
      if (isFmaCombinable(definingOp)) {
        auto op_mul = llvm::dyn_cast<mlir::arith::MulFOp>(definingOp);
        auto mul_inputs = op_mul->getOperands();
        // RankedTensorType ty = op->getOperands()[1].getType().template
        // dyn_cast<RankedTensorType>();
        RankedTensorType ty =
            llvm::dyn_cast<RankedTensorType>(op->getOperands()[1].getType());
        if (!ty)
          return mlir::failure();
        Type ty_t = RankedTensorType::get(ty.getShape(), ty.getElementType());
        FloatAttr constantAttr = FloatAttr::get(ty.getElementType(), -1.0);
        auto ConstOp = rewriter.create<triton::SplatOp>(
            op->getLoc(), ty_t,
            rewriter.create<arith::ConstantOp>(op->getLoc(), constantAttr));
        auto MulOp = rewriter.create<mlir::arith::MulFOp>(
            op->getLoc(), ty_t, op->getOperands()[1], ConstOp);
        auto FmaOp = rewriter.create<mlir::math::FmaOp>(
            op->getLoc(), ty_t, mul_inputs[0], mul_inputs[1], MulOp);
        rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(
            op, FmaOp.getA(), FmaOp.getB(), FmaOp.getC());
        return mlir::success();
      }
    }

    if (auto definingOp = op->getOperands()[1].getDefiningOp()) {
      if (isFmaCombinable(definingOp)) {
        auto op_mul = llvm::dyn_cast<mlir::arith::MulFOp>(definingOp);
        auto mul_inputs = op_mul->getOperands();
        // RankedTensorType ty = mul_inputs[1].getType().template
        // dyn_cast<RankedTensorType>();
        RankedTensorType ty =
            llvm::dyn_cast<RankedTensorType>(mul_inputs[1].getType());
        if (!ty)
          return mlir::failure();
        Type ty_t = RankedTensorType::get(ty.getShape(), ty.getElementType());
        FloatAttr constantAttr = FloatAttr::get(ty.getElementType(), -1.0);
        auto ConstOp = rewriter.create<triton::SplatOp>(
            op->getLoc(), ty_t,
            rewriter.create<arith::ConstantOp>(op->getLoc(), constantAttr));
        auto MulOp = rewriter.create<mlir::arith::MulFOp>(
            op->getLoc(), ty_t, mul_inputs[1], ConstOp);
        auto FmaOp = rewriter.create<mlir::math::FmaOp>(
            op->getLoc(), ty_t, mul_inputs[0], MulOp, op->getOperands()[0]);
        rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(
            op, FmaOp.getA(), FmaOp.getB(), FmaOp.getC());
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

bool getConstOpValue(arith::ConstantOp &constOp, int64_t &value) {
  auto valueAttr = constOp.getValue();
  if (!valueAttr)
    return false;
  auto denseAttr = dyn_cast<DenseIntElementsAttr>(valueAttr);
  if (!denseAttr)
    return false;
  auto valueInt = denseAttr.getSplatValue<APInt>();
  value = valueInt.getSExtValue();
  return true;
}

bool checkAddRem(arith::ConstantOp &constOp, Operation *op, int64_t &divValue) {
  auto remsiOp = dyn_cast<arith::RemSIOp>(op);
  if (!remsiOp)
    return false;
  int64_t addValue = 0;
  if (!getConstOpValue(constOp, addValue))
    return false;
  // RemSIOp
  auto remsiRhsOp = remsiOp.getRhs().getDefiningOp();
  if (!remsiRhsOp)
    return false;
  auto remsiRhsConstOp = dyn_cast<arith::ConstantOp>(remsiRhsOp);
  if (!remsiRhsConstOp)
    return false;
  int64_t remsiValue = 0;
  if (!getConstOpValue(remsiRhsConstOp, remsiValue))
    return false;
  if (addValue + remsiValue <= divValue && addValue + remsiValue >= 0 &&
      remsiValue > 0 && divValue > 0)
    return true;
  return false;
}

// y = (const_a + x % const_b) / const_c
// if we can infer that const_a + const_b <= const_c, then we can simplify y to
// 0
class FoldRemAddDivPattern : public mlir::RewritePattern {
public:
  FoldRemAddDivPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::DivSIOp::getOperationName(), 1,
                             context) {}

  mlir::LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto divSIOp = dyn_cast<arith::DivSIOp>(op);
    if (!divSIOp)
      return failure();
    auto ty = dyn_cast<RankedTensorType>(divSIOp.getType());
    if (!ty)
      return failure();
    // TODO(): support more type, only int now
    if (!ty.getElementType().isInteger())
      return failure();
    // TODO(): support shape size > 1
    if (ty.getShape().size() != 1)
      return failure();
    auto divRhsDefOp = divSIOp.getRhs().getDefiningOp();
    if (!divRhsDefOp)
      return failure();
    auto divRhsConstOp = dyn_cast<arith::ConstantOp>(divRhsDefOp);
    if (!divRhsConstOp)
      return failure();
    int64_t divValue = 0;
    if (!getConstOpValue(divRhsConstOp, divValue))
      return failure();
    auto divLhsDefOp = divSIOp.getLhs().getDefiningOp();
    if (!divLhsDefOp)
      return failure();
    // AddIOp
    auto divLhsAddOp = dyn_cast<arith::AddIOp>(divLhsDefOp);
    if (!divLhsAddOp)
      return failure();
    auto addLhsDefOp = divLhsAddOp.getLhs().getDefiningOp();
    if (!addLhsDefOp)
      return failure();
    auto addRhsDefOp = divLhsAddOp.getRhs().getDefiningOp();
    if (!addRhsDefOp)
      return failure();
    // AddIOp lhs or rhs is const
    auto addLhsConstOp = dyn_cast<arith::ConstantOp>(addLhsDefOp);
    auto addRhsConstOp = dyn_cast<arith::ConstantOp>(addRhsDefOp);
    if (addLhsConstOp || addRhsConstOp) {
      if (addLhsConstOp) {
        // const_a + x % const_b
        if (checkAddRem(addLhsConstOp, addRhsDefOp, divValue)) {
          IntegerAttr constantAttr = IntegerAttr::get(ty.getElementType(), 0);
          auto newConstant = rewriter.create<triton::SplatOp>(
              divSIOp.getLoc(), ty,
              rewriter.create<arith::ConstantOp>(divSIOp.getLoc(),
                                                 constantAttr));
          divSIOp.replaceAllUsesWith(newConstant.getResult());
          divSIOp.erase();
          return success();
        } else {
          return failure();
        }
      } else if (addRhsConstOp) {
        // x % const_b + const_a
        if (checkAddRem(addRhsConstOp, addLhsDefOp, divValue)) {
          IntegerAttr constantAttr = IntegerAttr::get(ty.getElementType(), 0);
          auto newConstant = rewriter.create<triton::SplatOp>(
              divSIOp.getLoc(), ty,
              rewriter.create<arith::ConstantOp>(divSIOp.getLoc(),
                                                 constantAttr));
          divSIOp.replaceAllUsesWith(newConstant.getResult());
          divSIOp.erase();
          return success();
        } else {
          return failure();
        }
      }
    }
    return failure();
  }
};
#endif

} // anonymous namespace

class CombineOpsPass : public impl::TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<CombineDotAddIPattern>(context);
    patterns.add<CombineDotAddFPattern>(context);
    patterns.add<CombineSelectMaskedLoadPattern>(context);
#ifdef USE_MACA
    if (!std::getenv("TRITON_DISABLE_COMBINE_ADD_PTR_PASS")) {
      patterns.add<CombineAddPtrPattern>(context);
    }
    // fma
    if (!std::getenv("TRITON_DISABLE_OP_FUSION_PASS")) {
      patterns.add<CombineAddfmulfPattern>(context);
      patterns.add<CombineSubfmulfPattern>(context);
    }
    if (!std::getenv("TRITON_DISABLE_FOLD_REM_ADD_DIV_PASS")) {
      patterns.add<FoldRemAddDivPattern>(context);
    }
#else
    patterns.add<CombineAddPtrPattern>(context);
#endif
    patterns.add<CombineBroadcastMulReducePattern>(context);
    patterns.add<CombineReshapeReducePatterns>(context);
    patterns.add<RankedReduceDescriptorLoads>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir::triton
