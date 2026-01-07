//===------------------- LinalgToMKPass.cpp -------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/LinalgToMK/LinalgToMK.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "linalg-to-mk"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LINALGTOMK
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class LinalgToMKPass : public triton::impl::LinalgToMKBase<LinalgToMKPass> {
  using LinalgToMKBase<LinalgToMKPass>::LinalgToMKBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, mk::MagicKernelDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    {
      // Fusion, identity reduction init, etc. Other ops which need to decompose
      // into multiple integer type operators also are converted.
      RewritePatternSet fusionPatterns(&getContext());
      triton::populateLinalgToMKPreProcessPatterns(fusionPatterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(fusionPatterns)))) {
        signalPassFailure();
      }
    }

    {
      RewritePatternSet typePatterns(&getContext());
      triton::populateLinalgToMKTypeConversionPatterns(typePatterns,
                                                       precisionPriority);
      if (failed(applyPatternsGreedily(moduleOp, std::move(typePatterns)))) {
        signalPassFailure();
      }
    }

    {
      // Layout transformation, and other canonicalization
      RewritePatternSet canonicalizePatterns(&getContext());
      triton::populateLinalgToMKCanonicalizationPatterns(canonicalizePatterns,
                                                         precisionPriority);
      if (failed(applyPatternsGreedily(moduleOp,
                                       std::move(canonicalizePatterns)))) {
        signalPassFailure();
      }
    }

    {
      RewritePatternSet shapePatterns(&getContext());
      triton::populateLinalgToMKShapeCanonicalizationPatterns(
          shapePatterns, precisionPriority);
      if (failed(applyPatternsGreedily(moduleOp, std::move(shapePatterns)))) {
        signalPassFailure();
      }
    }

    {
      // Target dependent conversion patterns
      RewritePatternSet patterns(&getContext());
      ConversionTarget target(getContext());
      target.addLegalDialect<
          func::FuncDialect, arith::ArithDialect, math::MathDialect,
          linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
          tensor::TensorDialect, bufferization::BufferizationDialect,
          memref::MemRefDialect, mk::MagicKernelDialect>();
      target.addDynamicallyLegalOp<linalg::ReduceOp>([&](linalg::ReduceOp op) {
        auto regionBlock = op.getBody();
        auto reduceOps = llvm::map_to_vector(regionBlock->without_terminator(),
                                             [](Operation &op) { return &op; });
        if (reduceOps.size() != 1)
          return true;
        // TODO: Config according backend
        // TODO: Optimize for i1 reduction. i1 reduction is not supported
        // because memref.subviews may cause the offset to be inside the byte.
        auto inputType =
            cast<RankedTensorType>(op.getInputs().front().getType());

        // NOTE: Assume has done integer to float conversion
        return !isReduceToElementWiseOpAndTypeSupportedByTarget(
            reduceOps.front(), inputType.getElementType(),
            inputType.getNumElements(), inputType.getRank());
      });

      // Reduce op conversion will generate arith/math tensor type op
      target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
          [](Operation *op) {
            // Lower dense constant to linalg.fill
            if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
              if (!isa<RankedTensorType>(constOp.getResult().getType())) {
                return true;
              }

              if (auto denseAttr =
                      dyn_cast<DenseElementsAttr>(constOp.getValue())) {
                if (isa<FloatType, IntegerType>(denseAttr.getElementType())) {
                  return false;
                }
              }
              return true;
            }

            bool operateOnTensors =
                llvm::all_of(op->getOperandTypes(), [](Type type) {
                  return isa<RankedTensorType>(type);
                });

            return !operateOnTensors;
          });

      triton::populateLinalgToMKConversionPatterns(patterns);
      // FIXME: Fixed pass pipeline order to avoid repeatedly adding
      // ElementwiseToLinalg patterns
      linalg::populateElementwiseToLinalgConversionPatterns(patterns);
      if (failed(
              applyPartialConversion(moduleOp, target, std::move(patterns)))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLinalgToMKPass() {
  return std::make_unique<LinalgToMKPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createLinalgToMKPass(LinalgToMKOptions &options) {
  return std::make_unique<LinalgToMKPass>(options);
}
