//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. ALL rights reserved.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation. All rights
// reserved. SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TypeConverter.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-arith-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONARITHTOLINALG
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class TritonArithToLinalgPass
    : public triton::impl::TritonArithToLinalgBase<TritonArithToLinalgPass> {
  using TritonArithToLinalgBase<
      TritonArithToLinalgPass>::TritonArithToLinalgBase;

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

  // Add additional I32 arguments to represent:
  // - num_programs, 3 in total, one for each axis of the launch grid
  // - program_id, 3 in total, one for each axis of the launch grid
  static void addProgramInfo(triton::FuncOp func) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add empty attributes for each new argument if needed
    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    // Add the corresponding arguments to function body
    for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

  LogicalResult applyTensorConcatDecomposition() {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    tensor::populateDecomposeTensorConcatPatterns(patterns);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return failure();
    }
    return success();
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect, mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      populateTritonArithToLinalgCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect, tts::TritonStructuredDialect>();

    target.addLegalOp<ModuleOp>();

    target.addLegalOp<triton::FuncOp, triton::ReturnOp, LLVM::PtrToIntOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          // Lower dense constant to linalg.fill
          if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            if (!isa<RankedTensorType>(constOp.getResult().getType())) {
              return true;
            }

            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constOp.getValue())) {
              if (denseAttr.isSplat() &&
                  isa<FloatType, IntegerType>(denseAttr.getElementType())) {
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

    if (pidsToFuncArgs) {
      target.addIllegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();
    }

    // PrintOp must be converted (generates LLVM ops)
    target.addIllegalOp<triton::PrintOp>();
    target.addLegalOp<LLVM::GlobalOp, LLVM::AddressOfOp, LLVM::LLVMFuncOp,
                      LLVM::CallOp>();

    if (addptrToLinalg) {
      target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
        return !isa<ShapedType>(op.getResult().getType());
      });
    }

    // target.addDynamicallyLegalOp<triton::BitcastOp>(
    //     [this](triton::BitcastOp op) {
    //       if (!tensorPtrToLinalg) {
    //         return triton::isPtrTypeLike(op.getType());
    //       } else {
    //         if (triton::isPtrTypeLike(op.getType())) {
    //           return !isa<ShapedType>(op.getType());
    //         }
    //         return false;
    //       }
    //     });

    // TODO: Might want to consolidate this flag with addptrToLinalg later.
    if (tensorPtrToLinalg) {
      target.addDynamicallyLegalOp<triton::LoadOp, triton::StoreOp,
                                   triton::IntToPtrOp>([](auto op) {
        return !isa<ShapedType>(op->getOperands()[0].getType());
      });
      populateTritonTensorPtrConversionPatterns(patterns);
    }

    if (!assertToCf) {
      target.addLegalOp<triton::AssertOp>();
    }

    triton::TritonLinalgTypeConverter converter;
    triton::populateTritonArithToLinalgConversionPatterns(
        pidsToFuncArgs, addptrToLinalg, assertToCf, patterns, converter);

    if (pidsToFuncArgs) {
      for (auto func : getOperation().getOps<triton::FuncOp>()) {
        addProgramInfo(func);
      }
    }

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    if (failed(applyTensorConcatDecomposition())) {
      signalPassFailure();
    }

    RewritePatternSet patterns1(&getContext());
    // ConversionTarget target1(getContext());
    triton::normalReduceConversionPatterns(patterns1);
    if (failed(
            applyPartialConversion(moduleOp, target, std::move(patterns1)))) {
      signalPassFailure();
    }

    // Convert tt.func and tt.return into func's counterparts
    if (ttToFuncFunc) {
      moduleOp.walk([&](triton::FuncOp func) {
        OpBuilder builder(func);

        auto name = func.getName();
        auto type = func.getFunctionType();

        SmallVector<DictionaryAttr> argAttrs, resAttrs;
        func.getAllArgAttrs(argAttrs);
        func.getAllResultAttrs(resAttrs);

        auto funcFunc =
            func::FuncOp::create(builder, func.getLoc(), name, type);
        funcFunc.setAllArgAttrs(argAttrs);
        funcFunc.setAllResultAttrs(resAttrs);

        auto &funcFuncBody = funcFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&funcFuncBody, map);

        for (Block &block : funcFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          // Only convert to func.return if the terminator is a tt.return.
          // Otherwise, we will accidentally convert cf.br ops which are also
          // considered terminators.
          if (isa<triton::ReturnOp>(term)) {
            builder.setInsertionPoint(term);
            func::ReturnOp::create(builder, func.getLoc(), term->getOperands());
            term->erase();
          }
        }
        func.erase();
      });
    }

    bool foldFill2Generic = true;
    if (foldFill2Generic) {
      moduleOp.walk([&](linalg::GenericOp genericOp) {
        OpBuilder builder(genericOp);
        SmallVector<Value> inputs = genericOp.getInputs();
        SmallVector<Value> outputs = genericOp.getOutputs();
        auto indexingMaps = genericOp.getIndexingMapsAttr();
        auto iteratorTypes = genericOp.getIteratorTypes();
        auto resultTypes = genericOp->getResultTypes();

        // Modify the input and index mapping
        bool modified = false;
        SmallVector<AffineMap> newIndexingMaps;
        for (auto map : indexingMaps.getAsRange<AffineMapAttr>()) {
          newIndexingMaps.push_back(map.getValue());
        }

        for (unsigned i = 0; i < inputs.size(); ++i) {
          if (auto fillOp = inputs[i].getDefiningOp<linalg::FillOp>()) {

            Value fillValue = fillOp.value();
            if (!fillValue.getType().isIntOrFloat() ||
                mlir::isa<ShapedType>(fillValue.getType()))
              continue;

            newIndexingMaps[i] = AffineMap::get(genericOp.getNumParallelLoops(),
                                                0, {}, builder.getContext());
            inputs[i] = fillOp.getOperand(0);
            modified = true;
          }
        }
        if (!modified)
          return;

        builder.setInsertionPoint(genericOp);
        auto newGenericOp = linalg::GenericOp::create(
            builder, genericOp.getLoc(), resultTypes, inputs, outputs,
            builder.getAffineMapArrayAttr(newIndexingMaps), iteratorTypes,
            genericOp.getDocAttr(), genericOp.getLibraryCallAttr());

        IRMapping mapping;
        genericOp.getRegion().cloneInto(&newGenericOp.getRegion(), mapping);

        newGenericOp.getRegion().walk([&](linalg::YieldOp yieldOp) {
          builder.setInsertionPoint(yieldOp);
          auto newYield = linalg::YieldOp::create(builder, yieldOp.getLoc(),
                                                  yieldOp.getOperands());
          yieldOp->replaceAllUsesWith(newYield);
          yieldOp->erase();
        });

        for (auto result :
             llvm::zip(genericOp->getResults(), newGenericOp->getResults())) {
          std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
        }

        if (genericOp->use_empty()) {
          builder.setInsertionPointAfter(newGenericOp);
          genericOp->erase();
        } else {
          newGenericOp->emitError("Failed to replace all uses");
        }
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonArithToLinalgPass(bool tensorPtrToLinalg) {
  TritonArithToLinalgOptions options;
  options.tensorPtrToLinalg = tensorPtrToLinalg;
  return std::make_unique<TritonArithToLinalgPass>(options);
}
