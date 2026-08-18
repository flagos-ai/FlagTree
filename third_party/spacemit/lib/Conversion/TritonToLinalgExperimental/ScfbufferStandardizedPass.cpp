//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ScfbufferStandardized.h"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_SCFBUFFERSTANDARDIZED
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct ConvertFillI1ToI8 : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    Location loc = fillOp.getLoc();
    Value input = fillOp.getInputs()[0];
    Value output = fillOp.getOutputs()[0];
    ShapedType outputType = cast<ShapedType>(output.getType());

    if (!input.getType().isInteger(1) ||
        !outputType.getElementType().isInteger(1)) {
      return failure();
    }

    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < outputType.getRank(); ++i) {
      if (outputType.isDynamicDim(i)) {
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, output, i));
      }
    }

    Type i8Type = rewriter.getI8Type();
    auto tensorI8Type = RankedTensorType::get(outputType.getShape(), i8Type);
    Value emptyI8 =
        tensor::EmptyOp::create(rewriter, loc, tensorI8Type, dynSizes);

    Value extendedInput = arith::ExtUIOp::create(rewriter, loc, i8Type, input);

    auto newFillOp = linalg::FillOp::create(
        rewriter, loc, ValueRange{extendedInput}, ValueRange{emptyI8});
    Value filledTensor = newFillOp.getResult(0);

    Value emptyI1 =
        tensor::EmptyOp::create(rewriter, loc, outputType, dynSizes);

    Value zeroI8 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(i8Type, 0));

    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
        outputType.getRank(), rewriter.getContext());

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{filledTensor},
        /*outputs=*/ValueRange{emptyI1},
        /*indexingMaps=*/ArrayRef<AffineMap>{identityMap, identityMap},
        /*iteratorTypes=*/
        SmallVector<utils::IteratorType>(outputType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value cmp = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                            args[0], zeroI8);
          linalg::YieldOp::create(b, loc, cmp);
        });

    rewriter.replaceOp(fillOp, genericOp.getResult(0));
    return success();
  }
};

struct ScfbufferStandardizedPass
    : public triton::impl::ScfbufferStandardizedBase<
          ScfbufferStandardizedPass> {
public:
  void runOnOperation() override {
    Operation *module = getOperation();
    module->walk([&](scf::ForOp forOp) { processForOp(forOp); });

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertFillI1ToI8>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  void processForOp(scf::ForOp forOp) {
    Block *body = forOp.getBody();
    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

    SmallVector<Value> iterArgs(forOp.getRegionIterArgs().begin(),
                                forOp.getRegionIterArgs().end());

    SmallVector<Value> yieldValues(yieldOp.getOperands().begin(),
                                   yieldOp.getOperands().end());

    if (iterArgs.size() != yieldValues.size())
      return;

    SmallVector<linalg::GenericOp> genericOps;
    SmallVector<int> resultIndices;
    SmallVector<int> argIndices;

    for (auto [idx, yieldVal] : llvm::enumerate(yieldValues)) {
      Operation *defOp = yieldVal.getDefiningOp();
      if (!defOp)
        continue;

      if (auto genericOp = dyn_cast<linalg::GenericOp>(defOp)) {
        auto results = genericOp->getResults();
        for (int resIdx = 0; resIdx < results.size(); ++resIdx) {
          if (results[resIdx] == yieldVal) {
            genericOps.push_back(genericOp);
            resultIndices.push_back(resIdx);
            argIndices.push_back(idx);
            break;
          }
        }
      }
    }

    for (auto [genericOp, resIdx, argIdx] :
         llvm::zip(genericOps, resultIndices, argIndices)) {
      Value iterArg = iterArgs[argIdx];
      SmallVector<Value> newOutputs = genericOp.getOutputs();
      if (resIdx < newOutputs.size()) {
        newOutputs[resIdx] = iterArg;
        OpBuilder builder(genericOp);
        genericOp.getOutputsMutable().assign(newOutputs);
      }
    }

    DenseMap<Value, Value> replacementMap;

    for (Operation &op : body->getOperations()) {
      auto genericOp = dyn_cast<linalg::GenericOp>(&op);
      if (!genericOp)
        continue;

      auto it = llvm::find(genericOps, genericOp);
      if (it == genericOps.end())
        continue;

      int pos = std::distance(genericOps.begin(), it);
      int resIdx = resultIndices[pos];
      int argIdx = argIndices[pos];

      Value iterArg = iterArgs[argIdx];
      Value newResult = genericOp.getResult(resIdx);

      replacementMap[iterArg] = newResult;

      SmallVector<OpOperand *> usesToReplace;
      for (OpOperand &use : iterArg.getUses()) {
        Operation *user = use.getOwner();
        if (user != genericOp && user->getBlock() == body &&
            genericOp->isBeforeInBlock(user)) {
          usesToReplace.push_back(&use);
        }
      }

      for (OpOperand *use : usesToReplace) {
        if (auto userGeneric = dyn_cast<linalg::GenericOp>(use->getOwner())) {
          auto operandNum = use->getOperandNumber();
          if (operandNum >= userGeneric.getInputs().size() &&
              operandNum < userGeneric.getInputs().size() +
                               userGeneric.getOutputs().size()) {
            int outputIdx = operandNum - userGeneric.getInputs().size();
            SmallVector<Value> newOutputs = userGeneric.getOutputs();
            newOutputs[outputIdx] = newResult;
            OpBuilder builder(userGeneric);
            userGeneric.getOutputsMutable().assign(newOutputs);
          }
        }
      }
    }

    SmallVector<Value> newYieldValues;
    bool changed = false;
    for (auto value : yieldOp.getOperands()) {
      if (replacementMap.count(value)) {
        newYieldValues.push_back(replacementMap[value]);
        changed = true;
      } else {
        newYieldValues.push_back(value);
      }
    }

    if (changed) {
      OpBuilder builder(yieldOp);
      scf::YieldOp::create(builder, yieldOp.getLoc(), newYieldValues);
      yieldOp.erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createScfbufferStandardizedPass() {
  return std::make_unique<ScfbufferStandardizedPass>();
}
