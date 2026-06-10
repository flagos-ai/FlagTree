//===----------------------------------------------------------------------===//
//
// EVAS Triton arithmetic to linalg conversion.
//
// This mirrors triton-shared's triton-arith-to-linalg pass, but lowers the
// tensor elementwise arithmetic needed by EVAS to named linalg ops directly.
//
//===----------------------------------------------------------------------===//

#include "evas/Conversion/TritonToEvas/TritonToEvasPipeline.h"
#include "evas/Dialect/Linalg/IR/LinalgOpsExt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

template <typename TritonOp, typename LinalgNamedOp>
class TritonToLinalgNamedConverter : public OpConversionPattern<TritonOp> {
public:
  using OpConversionPattern<TritonOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TritonOp op, typename TritonOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto dstType = dyn_cast<RankedTensorType>(op.getType());
    if (!dstType)
      return failure();

    auto init = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), dstType.getShape(), dstType.getElementType());
    auto namedOp = rewriter.create<LinalgNamedOp>(
        op.getLoc(), op.getType(), adaptor.getOperands(), ValueRange(init),
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, namedOp.getResults());
    return success();
  }
};

template <typename ArithCastOp>
class ArithCastToLinalgCastConverter
    : public OpConversionPattern<ArithCastOp> {
public:
  using OpConversionPattern<ArithCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithCastOp op, typename ArithCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto dstType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!dstType)
      return failure();

    auto init = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), dstType.getShape(), dstType.getElementType());
    auto castOp = rewriter.create<linalg::CastOp>(
        op.getLoc(), TypeRange(dstType), adaptor.getOperands()[0], init);
    rewriter.replaceOp(op, castOp.getResults());
    return success();
  }
};

void populateEvasElementwiseToLinalgPatterns(RewritePatternSet &patterns) {
  patterns
      .add<TritonToLinalgNamedConverter<arith::AddFOp, linalg::AddOp>,
           TritonToLinalgNamedConverter<arith::AddIOp, linalg::AddOp>,
           TritonToLinalgNamedConverter<arith::MulFOp, linalg::MulOp>,
           TritonToLinalgNamedConverter<arith::MulIOp, linalg::MulOp>,
           ArithCastToLinalgCastConverter<arith::ExtFOp>,
           ArithCastToLinalgCastConverter<arith::TruncFOp>>(
          patterns.getContext(), PatternBenefit(2));
}

class EvasTritonArithToLinalgPass
    : public PassWrapper<EvasTritonArithToLinalgPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EvasTritonArithToLinalgPass)

  EvasTritonArithToLinalgPass(bool tensorPtrToLinalg,
                              bool transposeReduceToRank0)
      : tensorPtrToLinalg(tensorPtrToLinalg),
        transposeReduceToRank0(transposeReduceToRank0) {}

  StringRef getArgument() const final { return "evas-triton-arith-to-linalg"; }
  StringRef getDescription() const final {
    return "Convert Triton arithmetic operations to linalg with EVAS named "
           "linalg elementwise lowering";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                mlir::triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      mlir::triton::populateTritonArithToLinalgCanonicalizationPatterns(
          patterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        tts::TritonStructuredDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<mlir::triton::FuncOp, mlir::triton::ReturnOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            if (!isa<RankedTensorType>(constOp.getResult().getType()))
              return true;
            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constOp.getValue())) {
              if (denseAttr.isSplat() &&
                  isa<FloatType, IntegerType>(denseAttr.getElementType()))
                return false;
            }
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return isa<RankedTensorType>(type);
              });
          return !operateOnTensors;
        });

    target.addIllegalOp<mlir::triton::GetProgramIdOp,
                        mlir::triton::GetNumProgramsOp>();
    target.addDynamicallyLegalOp<mlir::triton::AddPtrOp>(
        [](mlir::triton::AddPtrOp op) {
          return !isa<ShapedType>(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<mlir::triton::BitcastOp>(
        [this](mlir::triton::BitcastOp op) {
          if (!tensorPtrToLinalg)
            return mlir::triton::isPtrTypeLike(op.getType());
          if (mlir::triton::isPtrTypeLike(op.getType()))
            return !isa<ShapedType>(op.getType());
          return false;
        });

    if (tensorPtrToLinalg) {
      target.addDynamicallyLegalOp<mlir::triton::LoadOp,
                                   mlir::triton::StoreOp,
                                   mlir::triton::IntToPtrOp,
                                   mlir::triton::PtrToIntOp>([](auto op) {
        return !isa<ShapedType>(op->getOperands()[0].getType());
      });
      mlir::triton::populateTritonTensorPtrConversionPatterns(patterns);
    }

    populateEvasElementwiseToLinalgPatterns(patterns);
    mlir::triton::populateTritonArithToLinalgConversionPatterns(
        /*pidsToFuncArgs=*/true, /*addptrToLinalg=*/true,
        /*assertToCf=*/true, transposeReduceToRank0, patterns);

    addProgramInfo();

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    if (failed(applyTensorConcatDecomposition())) {
      signalPassFailure();
      return;
    }
    convertTritonFuncToFunc();
  }

private:
  static auto constexpr LAUNCH_GRID_RANK =
      mlir::triton::getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

  void addProgramInfo() {
    for (auto func : getOperation().getOps<mlir::triton::FuncOp>()) {
      OpBuilder b(func);
      auto origFuncType = func.getFunctionType();
      SmallVector<Type> newInputTypes(origFuncType.getInputs());
      newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());
      func.setFunctionType(
          b.getFunctionType(newInputTypes, origFuncType.getResults()));

      if (func.getAllArgAttrs()) {
        SmallVector<DictionaryAttr> newArgAttrs;
        func.getAllArgAttrs(newArgAttrs);
        newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
        func.setAllArgAttrs(newArgAttrs);
      }

      for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++)
        func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

  LogicalResult applyTensorConcatDecomposition() {
    RewritePatternSet patterns(&getContext());
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    return applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  void convertTritonFuncToFunc() {
    getOperation().walk([&](mlir::triton::FuncOp func) {
      OpBuilder builder(func);
      auto funcFunc = func::FuncOp::create(
          builder, func.getLoc(), func.getName(), func.getFunctionType());
      funcFunc.setVisibility(func.getVisibility());

      SmallVector<DictionaryAttr> argAttrs, resAttrs;
      func.getAllArgAttrs(argAttrs);
      func.getAllResultAttrs(resAttrs);
      funcFunc.setAllArgAttrs(argAttrs);
      funcFunc.setAllResultAttrs(resAttrs);

      IRMapping map;
      func.getBody().cloneInto(&funcFunc.getBody(), map);

      for (Block &block : funcFunc.getBody().getBlocks()) {
        Operation *term = block.getTerminator();
        if (isa<mlir::triton::ReturnOp>(term)) {
          builder.setInsertionPoint(term);
          func::ReturnOp::create(builder, func.getLoc(), term->getOperands());
          term->erase();
        }
      }
      func.erase();
    });
  }

  bool tensorPtrToLinalg;
  bool transposeReduceToRank0;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::evas::createEvasTritonArithToLinalgPass(
    bool tensorPtrToLinalg, bool transposeReduceToRank0) {
  return std::make_unique<EvasTritonArithToLinalgPass>(
      tensorPtrToLinalg, transposeReduceToRank0);
}
