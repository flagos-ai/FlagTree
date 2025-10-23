#ifndef TRITON_ADAPTER_ARGMINMAXCONVERTER_H
#define TRITON_ADAPTER_ARGMINMAXCONVERTER_H

#include "Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ConversionPatterns.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "triton-to-linalg"

#include "llvm/Support/Debug.h"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

template <typename T>
class ArgMinMaxBaseConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult matchTieBreakResult(Value currValue, Value currIndex,
                                    Value reduceValue, Value reduceIndex,
                                    mlir::Block::iterator &it,
                                    Value &tileBreakValue) const {
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto eqCmpOp = dyn_cast<arith::CmpFOp>(*it);
    if (eqCmpOp) {
      if (eqCmpOp.getPredicate() != arith::CmpFPredicate::OEQ ||
          currValue != eqCmpOp.getLhs() || reduceValue != eqCmpOp.getRhs()) {
        return failure();
      }
    }

    auto eqCmpIOp = dyn_cast<arith::CmpIOp>(*it++);
    if (eqCmpIOp) {
      if (eqCmpIOp.getPredicate() != arith::CmpIPredicate::eq ||
          currValue != eqCmpIOp.getLhs() || reduceValue != eqCmpIOp.getRhs()) {
        return failure();
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto sltCmpOp = dyn_cast<arith::CmpIOp>(*it++);
    if (!sltCmpOp || sltCmpOp.getPredicate() != arith::CmpIPredicate::slt ||
        currIndex != sltCmpOp.getLhs() || reduceIndex != sltCmpOp.getRhs()) {
      return failure();
    }

    // matching: %13 = arith.andi %11, %12 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto andOp = dyn_cast<arith::AndIOp>(*it++);

    Value cmpOp;
    if (eqCmpOp)
      cmpOp = eqCmpOp;
    else
      cmpOp = eqCmpIOp;

    if (!andOp || andOp.getLhs() != cmpOp || andOp.getRhs() != sltCmpOp) {
      return failure();
    }

    tileBreakValue = andOp;
    return success();
  }

  LogicalResult matchShouldUpdateValue(Value currValue, Value currIndex,
                                       Value reduceValue, Value reduceIndex,
                                       mlir::Block::iterator &it,
                                       Value &shouldUpdate) const {
    Value tieResult;
    if (failed(matchTieBreakResult(currValue, currIndex, reduceValue,
                                   reduceIndex, it, tieResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Tie break result match failed\n");
      return failure();
    }

    Value comparisonResult;
    if (failed(T::matchComparisonResult(currValue, currIndex, reduceValue,
                                        reduceIndex, it, comparisonResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Comparison result match failed\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto orOp = dyn_cast<arith::OrIOp>(*it++);
    if (!orOp || orOp.getLhs() != comparisonResult ||
        orOp.getRhs() != tieResult) {
      return failure();
    }

    shouldUpdate = orOp;
    return success();
  }

  Value getInitTensor(ConversionPatternRewriter &rewriter,
                      ArrayRef<int64_t> shape, Value fillValue,
                      Location loc) const {
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, shape, fillValue.getType());
    return rewriter
        .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                ValueRange{initTensor})
        .result();
  }

public:
  ArgMinMaxBaseConverter(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult match(triton::ReduceOp op) const override final {
    if (op.getBody()->getNumArguments() != 4) {
      return failure();
    }

    auto block = op.getBody();
    auto ops = block->without_terminator();

    Value currValue = block->getArgument(0);
    Value currIndex = block->getArgument(1);
    Value reduceValue = block->getArgument(2);
    Value reduceIndex = block->getArgument(3);

    auto opsIt = ops.begin();
    Value shouldUpdate;
    if (failed(matchShouldUpdateValue(currValue, currIndex, reduceValue,
                                      reduceIndex, opsIt, shouldUpdate))) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto valueSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (!valueSelectOp || valueSelectOp.getCondition() != shouldUpdate ||
        currValue != valueSelectOp.getTrueValue() ||
        reduceValue != valueSelectOp.getFalseValue()) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto indexSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (indexSelectOp) {
      if (indexSelectOp.getCondition() != shouldUpdate ||
          currIndex != indexSelectOp.getTrueValue() ||
          reduceIndex != indexSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }
    if (!indexSelectOp || indexSelectOp.getCondition() != shouldUpdate ||
        currIndex != indexSelectOp.getTrueValue() ||
        reduceIndex != indexSelectOp.getFalseValue()) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto termOp = dyn_cast<triton::ReduceReturnOp>(*opsIt++);
    if (!(termOp && termOp == block->getTerminator() &&
          termOp.getOperands() ==
              ArrayRef<Value>{valueSelectOp, indexSelectOp})) {
      return failure();
    }
    return success();
  }

  void rewrite(triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override final {
    auto loc = op.getLoc();
    auto elemTypes = op.getElementTypes();

    auto valueType = elemTypes[0];
    // tl.argmin reorder
    auto block = op.getBody();
    if (isa<mlir::FloatType>(valueType)) {
      arith::CmpFOp cmpFOp;
      block->walk([&](arith::CmpFOp cmpOp) {
        auto pred = cmpOp.getPredicate();
        if (pred == arith::CmpFPredicate::OEQ ||
            pred == arith::CmpFPredicate::ONE ||
            pred == arith::CmpFPredicate::UEQ ||
            pred == arith::CmpFPredicate::UNE) {
          return WalkResult::advance();
        } else if (pred == arith::CmpFPredicate::OGT ||
                   pred == arith::CmpFPredicate::OLT ||
                   pred == arith::CmpFPredicate::UGT ||
                   pred == arith::CmpFPredicate::ULT) {
          cmpFOp = cmpOp;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      cmpFOp->moveBefore(block, block->getOperations().begin());
    } else if (isa<mlir::IntegerType>(valueType)) {
      arith::CmpIOp cmpIOp;
      block->walk([&](arith::CmpIOp cmpOp) {
        auto pred = cmpOp.getPredicate();
        if (pred == arith::CmpIPredicate::eq ||
            pred == arith::CmpIPredicate::ne) {
          return WalkResult::advance();
        } else if (pred == arith::CmpIPredicate::sgt ||
                   pred == arith::CmpIPredicate::slt ||
                   pred == arith::CmpIPredicate::ugt ||
                   pred == arith::CmpIPredicate::ult) {
          if (cmpOp.getLhs() == block->getArgument(0) &&
              cmpOp.getRhs() == block->getArgument(2)) {
            cmpIOp = cmpOp;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      cmpIOp->moveBefore(block, block->getOperations().begin());
    }

    TypedAttr valueAttr;
    if (isa<mlir::FloatType>(valueType)) {
      valueAttr = rewriter.getFloatAttr(valueType, T::getBaseReductionValue());
    } else if (isa<mlir::IntegerType>(valueType)) {
      // TODO: support other type of int
      valueAttr =
          rewriter.getIntegerAttr(valueType, T::getBaseReductionIntValue());
    }

    auto valuesAccBaseVal =
        rewriter.create<arith::ConstantOp>(loc, valueType, valueAttr);

    auto indexType = elemTypes[1];
    auto indicesAccBaseVal = rewriter.create<arith::ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, -1));

    auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
    const auto isScalarReduce = valueResultType == nullptr;
    SmallVector<int64_t> reductionResultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(valueResultType.getShape())};

    SmallVector<Value> outputs{
        getInitTensor(rewriter, reductionResultShape, valuesAccBaseVal, loc),
        getInitTensor(rewriter, reductionResultShape, indicesAccBaseVal, loc)};

    auto linalgOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getOperands(), outputs,
        SmallVector<int64_t>{adaptor.getAxis()},
        [&](OpBuilder &b, Location loc, ValueRange inputs) {
          assert(inputs.size() == 4);

          auto tritonReduceBlock = op.getBody();
          IRMapping mapping;
          mapping.map(tritonReduceBlock->getArguments(), inputs);

          for (auto &op : tritonReduceBlock->without_terminator()) {
            b.clone(op, mapping);
          }

          auto tritonYield = tritonReduceBlock->getTerminator();
          auto results =
              llvm::map_to_vector(tritonYield->getOperands(), [&](Value val) {
                return mapping.lookup(val);
              });
          b.create<linalg::YieldOp>(loc, results);
        });

    // before we rewrite the argmax reduce op, we know it has return value
    // so addReduceWithIndexAttrIfNeeded won't fail
    // but ignoring it will lead to compiling failure
    auto logicalResult = addReduceWithIndexAttrIfNeeded(rewriter, linalgOp);

    if (isScalarReduce) {
      SmallVector<Value> reduceResults{
          rewriter.create<tensor::ExtractOp>(
              loc, valueType, linalgOp.getResults()[0], ValueRange{}),
          rewriter.create<tensor::ExtractOp>(
              loc, indexType, linalgOp.getResults()[1], ValueRange{})};
      rewriter.replaceOp(op, reduceResults);
    } else {
      rewriter.replaceOp(op, linalgOp);
    }
  }
};

class ArgMinConverter : public ArgMinMaxBaseConverter<ArgMinConverter> {
public:
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult);

  static float getBaseReductionValue();

  static int8_t getBaseReductionIntValue();

  ArgMinConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

class ArgMaxConverter : public ArgMinMaxBaseConverter<ArgMaxConverter> {
public:
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult);

  static float getBaseReductionValue();

  static int8_t getBaseReductionIntValue();

  ArgMaxConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

} // namespace TTOpConverters

#endif