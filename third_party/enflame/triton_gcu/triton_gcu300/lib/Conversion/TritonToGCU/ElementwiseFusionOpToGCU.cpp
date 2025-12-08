/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <map>
#include <string>

#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "PatternTritonGPUOpToGCU.h"

#include "Analysis/FirstLastUserAnalysis.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "TritonGCUToGCU/TritonGCUToGCUUtils.h"
#include "Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

static constexpr unsigned vectorizationMaxLength = 16384;

bool hasBuiltinImpl(Operation *op) {
  auto isArithOp =
      isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
          arith::MulIOp, arith::MulFOp, arith::DivSIOp, arith::DivUIOp,
          arith::DivFOp, arith::RemSIOp, arith::RemUIOp, arith::MaxSIOp,
          arith::MaxUIOp, arith::MaxNumFOp, arith::MaximumFOp, arith::MinSIOp,
          arith::MinUIOp, arith::MinNumFOp, arith::MinimumFOp, arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
          arith::ShRUIOp, arith::NegFOp, arith::ExtFOp, arith::TruncFOp,
          arith::ExtSIOp, arith::ExtUIOp, arith::SIToFPOp, arith::UIToFPOp,
          arith::FPToUIOp, arith::FPToSIOp>(op);
  auto isMathOp =
      isa<math::AbsFOp, math::AbsIOp, math::CosOp, math::CoshOp, math::AcosOp,
          math::AcoshOp, math::SinOp, math::SinhOp, math::AsinOp, math::TanOp,
          math::TanhOp, math::AtanOp, math::Atan2Op, math::AtanhOp,
          math::SqrtOp, math::RsqrtOp, math::CbrtOp, math::FPowIOp,
          math::IPowIOp, math::PowFOp, math::ErfOp, math::ExpOp, math::Exp2Op,
          math::ExpM1Op, math::CeilOp, math::FloorOp, math::LogOp,
          math::Log10Op, math::Log1pOp, math::Log2Op, math::RoundOp,
          math::RoundEvenOp>(op);

  if (auto ewOp = dyn_cast<triton::ExternElementwiseOp>(op)) {
    return (
        ewOp.getSymbol() != "__nv_ffs" && ewOp.getSymbol() != "__nv_isnanf" &&
        ewOp.getSymbol() != "__nv_isinff" &&
        ewOp.getSymbol() != "__nv_finitef" && ewOp.getSymbol() != "__nv_fmodf");
  }
  return isArithOp || isMathOp;
}

std::string getBuiltinOpSymbol(Operation *op) {
  if (isa<triton::ExternElementwiseOp>(op)) {
    auto ewOp = dyn_cast<triton::ExternElementwiseOp>(op);
    auto name = ewOp.getSymbol();
    if (name == "__nv_fmaxf") {
      return "maximumf";
    } else if (name == "__nv_fminf") {
      return "minimumf";
    } else if (name == "__nv_floorf") {
      return "floor";
    } else if (name == "__nv_min") {
      return "minsi";
    } else if (name == "__nv_max") {
      return "maxsi";
    } else if (name == "__nv_umin") {
      return "minui";
    } else if (name == "__nv_umax") {
      return "maxui";
    } else if (name == "__nv_powf") {
      return "powf";
    } else if (name == "__nv_powif") {
      return "fpowi";
    } else if (name == "__nv_log2f") {
      return "log2";
    } else if (name == "__nv_exp2f") {
      return "exp2";
    } else if (name == "__nv_atan2f") {
      return "atan2";
    } else if (name == "__nv_tanhf") {
      return "tanh";
    } else if (name == "__nv_erff") {
      return "erf";
    } else if (name == "__nv_sqrtf") {
      return "sqrt";
    } else if (name == "__nv_rsqrtf") {
      return "rsqrt";
    } else if (name == "__nv_rintf") {
      return "roundeven";
    } else {
      llvm_unreachable(
          ("unsupported extern elementwise: " + name).str().c_str());
    }
  } else {
    return op->getName().getStringRef().split('.').second.str();
  }
}

struct GCUElementwiseFusionOpLowering
    : SharedConversionPattern<triton::gcu::ElementwiseFusionRegionOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::ElementwiseFusionRegionOp op,
                  SharedConversionPattern::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    auto loc = op.getLoc();
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    auto totalNumElems =
        triton::gcu::getTotalElemsPerThread(op.getResultTypes().front());

    DenseSet<Type> elementTypeSet;
    SmallVector<Value> results;
    SmallVector<Value> outputs;
    for (auto type : op.getResultTypes()) {
      auto resultType =
          dyn_cast<MemRefType>(getTypeConverter()->convertType(type));
      auto result = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, resultType);
      results.push_back(result);
      auto elementTy = resultType.getElementType();
      elementTypeSet.insert(elementTy);
      if (elementTy.isInteger(1)) {
        outputs.emplace_back(rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(ArrayRef<int64_t>{totalNumElems},
                            rewriter.getIntegerType(8)),
            rewriter.create<mlir::gcu::PtrToMemRefOp>(
                loc,
                MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                                rewriter.getIntegerType(8)),
                rewriter.create<mlir::gcu::MemRefToPtrOp>(
                    loc,
                    mlir::gcu::PtrType::get(rewriter.getContext(), elementTy),
                    result)),
            0, ArrayRef<int64_t>{totalNumElems}, ArrayRef<int64_t>{1}));
      } else {
        outputs.emplace_back(rewriter.create<memref::ReinterpretCastOp>(
            loc, MemRefType::get(ArrayRef<int64_t>{totalNumElems}, elementTy),
            result, 0, ArrayRef<int64_t>{totalNumElems}, ArrayRef<int64_t>{1}));
      }
    }

    SmallVector<Value> inputs;
    SmallVector<Type> elementTypes;
    for (auto operand : adaptor.getOperands()) {
      auto operandType = operand.getType();
      if (isa<MemRefType>(operandType)) {
        auto elementTy = cast<MemRefType>(operandType).getElementType();
        elementTypes.push_back(elementTy);
        elementTypeSet.insert(elementTy);
        if (elementTy.isInteger(1)) {
          inputs.emplace_back(rewriter.create<memref::ReinterpretCastOp>(
              loc,
              MemRefType::get(ArrayRef<int64_t>{totalNumElems},
                              rewriter.getIntegerType(8)),
              rewriter.create<mlir::gcu::PtrToMemRefOp>(
                  loc,
                  MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                                  rewriter.getIntegerType(8)),
                  rewriter.create<mlir::gcu::MemRefToPtrOp>(
                      loc,
                      mlir::gcu::PtrType::get(rewriter.getContext(), elementTy),
                      operand)),
              0, ArrayRef<int64_t>{totalNumElems}, ArrayRef<int64_t>{1}));
        } else {
          inputs.emplace_back(rewriter.create<memref::ReinterpretCastOp>(
              loc, MemRefType::get(ArrayRef<int64_t>{totalNumElems}, elementTy),
              operand, 0, ArrayRef<int64_t>{totalNumElems},
              ArrayRef<int64_t>{1}));
        }
      } else {
        elementTypeSet.insert(operandType);
        elementTypes.push_back(operandType);
        inputs.push_back(operand);
      }
    }

    for (auto &o : op.getRegion().back().without_terminator()) {
      for (auto type : o.getResultTypes()) {
        auto elementTy = getTypeConverter()->convertType(
            cast<TensorType>(type).getElementType());
        if (!elementTy.isInteger(1)) {
          elementTypeSet.insert(elementTy);
        }
      }
    }

    unsigned maxBpe = 1;
    unsigned minBpe = targetInfo[GCU300].supportI64 ? 8 : 4;
    for (auto elementTy : elementTypeSet) {
      auto bpe = mlir::triton::gcu::getBpe(elementTy);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    unsigned numVacc = maxBpe / minBpe;
    assert(numVacc <= 4);
    (void)numVacc;
    unsigned vectorLength = targetInfo[GCU300].vaccSizeInBytes *
                            targetInfo[GCU300].preferVaccNum / maxBpe;

    // TODO(peng.tian) Remove after enable some optimization.
    if (llvm::all_of(
            adaptor.getOperands(),
            [](auto operand) { return isa<MemRefType>(operand.getType()); }) &&
        op.getRegion().hasOneBlock()) {
      auto &ops = op.getRegion().front().getOperations();
      if (totalNumElems > vectorizationMaxLength / maxBpe) {
        if (ops.size() == 2 && hasBuiltinImpl(&ops.front())) {
          SmallVector<Value, 4> builtinOperands;
          for (auto operand : ops.front().getOperands()) {
            builtinOperands.push_back(
                inputs[dyn_cast<BlockArgument>(operand).getArgNumber()]);
          }
          auto opName = getBuiltinOpSymbol(&ops.front());
          rewriter.create<gcu::BuiltinElementwiseOp>(
              loc, outputs[0], builtinOperands, rewriter.getStringAttr(opName));
          rewriter.replaceOp(op, results);
          return success();
        }
        if (ops.size() == 3) {
          auto &op0 = ops.front();
          auto &elementWiseOp = *std::next(ops.begin(), 1);
          if (isa<arith::ConstantOp, triton::SplatOp>(op0) &&
              isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
                  arith::MulIOp, arith::MulFOp, arith::DivSIOp, arith::DivUIOp,
                  arith::DivFOp, arith::MaxSIOp, arith::MaxUIOp,
                  arith::MaxNumFOp, arith::MaximumFOp, arith::MinSIOp,
                  arith::MinUIOp, arith::MinNumFOp, arith::MinimumFOp,
                  arith::AndIOp, math::FPowIOp, math::IPowIOp, math::PowFOp>(
                  &elementWiseOp) &&
              llvm::hasSingleElement(op0.getUsers()) &&
              *op0.user_begin() == &elementWiseOp) {
            SmallVector<Value, 4> builtinOperands;
            for (auto operand : elementWiseOp.getOperands()) {
              if (isa<BlockArgument>(operand)) {
                builtinOperands.push_back(
                    inputs[cast<BlockArgument>(operand).getArgNumber()]);
              } else {
                assert(operand.getDefiningOp() == &op0);
                if (auto constantOp = dyn_cast<arith::ConstantOp>(op0)) {
                  auto elementTy = getTypeConverter()->convertType(
                      dyn_cast<TensorType>(operand.getType()).getElementType());
                  Value builtinOperand = rewriter.create<arith::ConstantOp>(
                      loc, elementTy,
                      dyn_cast<DenseElementsAttr>(constantOp.getValue())
                          .getSplatValue<TypedAttr>());
                  if (elementTy.isInteger(1)) {
                    builtinOperand = rewriter.create<arith::ExtUIOp>(
                        loc, rewriter.getIntegerType(8), builtinOperand);
                  }
                  builtinOperands.push_back(builtinOperand);
                } else if (auto splatOp = dyn_cast<triton::SplatOp>(op0)) {
                  assert(isa<BlockArgument>(splatOp.getSrc()));
                  Value builtinOperand =
                      inputs[cast<BlockArgument>(splatOp.getSrc())
                                 .getArgNumber()];
                  if (splatOp.getSrc().getType().isInteger(1)) {
                    builtinOperand = rewriter.create<arith::ExtUIOp>(
                        loc, rewriter.getIntegerType(8), builtinOperand);
                  }
                  builtinOperands.push_back(builtinOperand);
                }
              }
            }
            auto opName = getBuiltinOpSymbol(&elementWiseOp);
            rewriter.create<gcu::BuiltinElementwiseOp>(
                loc, outputs[0], builtinOperands,
                rewriter.getStringAttr(opName));
            rewriter.replaceOp(op, results);
            return success();
          }
        }
      }
    }

    constexpr unsigned loopUnrollTime = 1;
    auto loopLimit = ceil<unsigned>(totalNumElems, vectorLength);
    auto loopCnt = loopUnrollTime > loopLimit ? loopLimit : loopUnrollTime;

    auto insertPoint = rewriter.saveInsertionPoint();

    SmallVector<IRMapping> operandMaps(loopCnt);
    SmallVector<Value> initValues;
    Value step;

    for (auto &o : op.getRegion().back().without_terminator()) {
      if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(o)) {
        auto startIdx = makeRangeOp.getStart();
        auto elementTy = makeRangeOp.getResult().getType().getElementType();
        Value start =
            rewriter.create<arith::ConstantIntOp>(loc, startIdx, elementTy)
                .getResult();
        if (!getSlicedAxies(makeRangeOp.getType()).empty()) {
          start = rewriter.create<arith::AddIOp>(
              loc,
              rewriter.create<arith::MulIOp>(
                  loc,
                  rewriter.create<arith::IndexCastOp>(
                      loc, elementTy,
                      getWarpIds(rewriter, loc, makeRangeOp.getType()).front()),
                  rewriter.create<arith::ConstantIntOp>(loc, totalNumElems,
                                                        elementTy)),
              start);
        }
        initValues.emplace_back(
            rewriter
                .create<gcu::VectorStepOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
                    start)
                .getResult());
      }
    }

    rewriter.create<scf::ForOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems),
        rewriter.create<arith::ConstantIndexOp>(loc, vectorLength * loopCnt),
        initValues,
        [&](OpBuilder &builder, Location loc, Value iter, ValueRange iterArgs) {
          SmallVector<Value> args(iterArgs);
          for (unsigned i = 0; i < loopCnt; ++i) {
            for (unsigned j = 0; j < inputs.size(); ++j) {
              if (isa<MemRefType>(inputs[j].getType())) {
                auto elementTy = elementTypes[j];
                if (elementTy.isInteger(1)) {
                  elementTy = builder.getIntegerType(8);
                }
                operandMaps[i].map(
                    op.getRegion().getArgument(j),
                    builder.create<vector::LoadOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        elementTy),
                        inputs[j],
                        ValueRange{builder.create<arith::AddIOp>(
                            loc,
                            builder.create<arith::ConstantIndexOp>(
                                loc, i * vectorLength),
                            iter)}));
              } else {
                operandMaps[i].map(op.getRegion().getArgument(j), inputs[j]);
              }
            }
          }
          for (unsigned i = 0; i < loopCnt; ++i) {
            unsigned argIndex = 0;
            for (auto &o : op.getRegion().back().without_terminator()) {
              if (auto bitcastOp = dyn_cast<triton::BitcastOp>(o)) {
                handleBitcastOp(bitcastOp, builder, operandMaps[i],
                                vectorLength);
              } else if (auto splatOp = dyn_cast<triton::SplatOp>(o)) {
                if (i == 0) {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.restoreInsertionPoint(insertPoint);
                  handleSplatOp(splatOp, builder, operandMaps[i], vectorLength);
                } else {
                  operandMaps[i].map(
                      splatOp.getResult(),
                      operandMaps[0].lookup(splatOp.getResult()));
                }
              } else if (auto constantOp = dyn_cast<arith::ConstantOp>(o)) {
                if (i == 0) {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.restoreInsertionPoint(insertPoint);
                  handleConstantOp(constantOp, builder, operandMaps[i],
                                   vectorLength);
                } else {
                  operandMaps[i].map(
                      constantOp.getResult(),
                      operandMaps[0].lookup(constantOp.getResult()));
                }
              } else if (auto externElementwiseOp =
                             dyn_cast<triton::ExternElementwiseOp>(o)) {
                handleExternElementwiseOp(externElementwiseOp, builder,
                                          operandMaps[i], vectorLength);
              } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(o)) {
                if (i == 0) {
                  auto elementTy =
                      makeRangeOp.getResult().getType().getElementType();
                  step = rewriter.create<vector::BroadcastOp>(
                      loc,
                      VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      elementTy),
                      rewriter.create<arith::ConstantIntOp>(loc, vectorLength,
                                                            elementTy));
                }
                operandMaps[i].map(makeRangeOp.getResult(), args[argIndex]);
                args[argIndex] =
                    rewriter.create<arith::AddIOp>(loc, args[argIndex], step);
                ++argIndex;
              } else {
                handleCommonOp(o, builder, operandMaps[i], vectorLength);
              }
            }
          }
          if (auto yieldOp = cast<triton::gcu::YieldOp>(
                  op.getRegion().back().getTerminator())) {
            for (unsigned i = 0; i < loopCnt; ++i) {
              for (unsigned j = 0; j < yieldOp.getNumOperands(); ++j) {
                auto v = operandMaps[i].lookup(yieldOp.getOperand(j));
                if (dyn_cast<VectorType>(v.getType())
                        .getElementType()
                        .isInteger(1)) {
                  OpBuilder::InsertionGuard guard(builder);
                  auto defOp = v.getDefiningOp();
                  assert(defOp);
                  builder.setInsertionPointAfter(defOp);
                  v = builder
                          .create<gcu::VectorConvertOp>(
                              loc,
                              VectorType::get(ArrayRef<int64_t>{vectorLength},
                                              builder.getIntegerType(8)),
                              v)
                          .getResult(0);
                }
                builder.create<vector::StoreOp>(
                    loc, v, outputs[j],
                    ValueRange{builder.create<arith::AddIOp>(
                        loc,
                        builder.create<arith::ConstantIndexOp>(
                            loc, i * vectorLength),
                        iter)});
              }
            }
            builder.create<scf::YieldOp>(loc, args);
          }
        });
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  void handleConstantOp(arith::ConstantOp op, OpBuilder &builder,
                        IRMapping &map, unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto elementTy = getTypeConverter()->convertType(
        dyn_cast<TensorType>(op.getType()).getElementType());
    auto vectorType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
    Value v;
    if (elementTy.isInteger(1)) {
      v = builder.create<vector::BroadcastOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(8)),
          builder.create<arith::ExtUIOp>(
              loc, builder.getIntegerType(8),
              builder.create<arith::ConstantOp>(
                  loc, elementTy,
                  dyn_cast<DenseElementsAttr>(op.getValue())
                      .getSplatValue<TypedAttr>())));
    } else {
      v = builder.create<vector::BroadcastOp>(
          loc, vectorType,
          builder.create<arith::ConstantOp>(
              loc, elementTy,
              dyn_cast<DenseElementsAttr>(op.getValue())
                  .getSplatValue<TypedAttr>()));
    }
    map.map(op.getResult(), v);
  }

  void handleSplatOp(triton::SplatOp op, OpBuilder &builder, IRMapping &map,
                     unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto elementTy = getTypeConverter()->convertType(
        dyn_cast<TensorType>(op.getType()).getElementType());
    Value v;
    if (elementTy.isInteger(1)) {
      v = builder
              .create<vector::BroadcastOp>(
                  loc,
                  VectorType::get(ArrayRef<int64_t>{vectorLength},
                                  builder.getIntegerType(8)),
                  builder.create<arith::ExtUIOp>(loc, builder.getIntegerType(8),
                                                 map.lookup(op.getSrc())))
              .getResult();
    } else {
      v = builder.create<vector::BroadcastOp>(
          loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
          map.lookup(op.getSrc()));
    }
    map.map(op.getResult(), v);
  }

  void handleBitcastOp(triton::BitcastOp op, OpBuilder &builder, IRMapping &map,
                       unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto vectorType = VectorType::get(
        ArrayRef<int64_t>{vectorLength},
        getTypeConverter()->convertType(
            dyn_cast<TensorType>(op.getType()).getElementType()));
    auto newOp = builder.create<arith::BitcastOp>(loc, vectorType,
                                                  map.lookup(op.getOperand()));
    map.map(op.getResult(), newOp.getResult());
  }

  void handleExternElementwiseOp(triton::ExternElementwiseOp op,
                                 OpBuilder &builder, IRMapping &map,
                                 unsigned vectorLength) const {
    SmallVector<Value, 4> operands;
    auto loc = op.getLoc();
    for (auto operand : op.getOperands()) {
      operands.push_back(map.lookup(operand));
    }
    auto symbol = op.getSymbol();
    Operation *newOp;
    if (symbol == "__nv_fmaxf") {
      newOp = builder.create<arith::MaximumFOp>(loc, operands.front().getType(),
                                                operands);
    } else if (symbol == "__nv_fminf") {
      newOp = builder.create<arith::MinimumFOp>(loc, operands.front().getType(),
                                                operands);
    } else if (symbol == "__nv_floorf") {
      newOp = builder.create<math::FloorOp>(loc, operands.front().getType(),
                                            operands);
    } else if (symbol == "__nv_min") {
      newOp = builder.create<arith::MinSIOp>(loc, operands.front().getType(),
                                             operands);
    } else if (symbol == "__nv_max") {
      newOp = builder.create<arith::MaxSIOp>(loc, operands.front().getType(),
                                             operands);
    } else if (symbol == "__nv_umin") {
      newOp = builder.create<arith::MinUIOp>(loc, operands.front().getType(),
                                             operands);
    } else if (symbol == "__nv_umax") {
      newOp = builder.create<arith::MaxUIOp>(loc, operands.front().getType(),
                                             operands);
    } else if (symbol == "__nv_powf") {
      newOp = builder.create<math::PowFOp>(loc, operands.front().getType(),
                                           operands);
    } else if (symbol == "__nv_powif") {
      newOp = builder.create<math::FPowIOp>(loc, operands.front().getType(),
                                            operands);
    } else if (symbol == "__nv_log2f") {
      newOp = builder.create<math::Log2Op>(loc, operands.front().getType(),
                                           operands);
    } else if (symbol == "__nv_exp2f") {
      newOp = builder.create<math::Exp2Op>(loc, operands.front().getType(),
                                           operands);
    } else if (symbol == "__nv_ffs") {
      newOp = builder.create<math::CtPopOp>(loc, operands.front().getType(),
                                            operands);
    } else if (symbol == "__nv_erff") {
      newOp = builder.create<math::ErfOp>(loc, operands.front().getType(),
                                          operands);
    } else if (symbol == "__nv_tanhf") {
      newOp = builder.create<math::TanhOp>(loc, operands.front().getType(),
                                           operands);
    } else if (symbol == "__nv_fdiv_rn") {
      auto resType = operands.front().getType();
      auto vectorType = dyn_cast<VectorType>(resType);
      auto elemType = vectorType.getElementType();
      auto constValue = builder.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vectorType,
                                      builder.getFloatAttr(elemType, 0.5)));

      auto div = builder.create<arith::DivFOp>(loc, resType, operands);
      newOp = builder.create<math::FloorOp>(
          loc, resType, builder.create<arith::AddFOp>(loc, div, constValue));
    } else if (symbol == "__nv_fdiv_rz") {
      auto resType = operands.front().getType();
      auto vectorType = dyn_cast<VectorType>(resType);
      auto elemType = vectorType.getElementType();
      auto zero = builder.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vectorType,
                                      builder.getFloatAttr(elemType, 0)));

      auto div = builder.create<arith::DivFOp>(loc, resType, operands);

      newOp = builder.create<arith::SelectOp>(
          loc,
          builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, div,
                                        zero),
          builder.create<math::FloorOp>(loc, resType, div),
          builder.create<math::CeilOp>(loc, resType, div));
    } else if (symbol == "__nv_fmodf") {
      auto resType = operands.front().getType();
      auto vectorType = dyn_cast<VectorType>(resType);
      auto elemType = vectorType.getElementType();
      auto zero = builder.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vectorType,
                                      builder.getFloatAttr(elemType, 0)));

      auto div = builder.create<arith::DivFOp>(loc, resType, operands);

      auto vfloor = builder.create<arith::SelectOp>(
          loc,
          builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, div,
                                        zero),
          builder.create<math::FloorOp>(loc, resType, div),
          builder.create<math::CeilOp>(loc, resType, div));

      newOp = builder.create<arith::SubFOp>(
          loc, operands[0],
          builder.create<arith::MulFOp>(loc, vfloor, operands[1]));
    } else if (symbol == "__nv_truncf") {
      auto resType = operands.front().getType();
      auto vectorType = dyn_cast<VectorType>(resType);
      auto elemType = vectorType.getElementType();
      auto zero = builder.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(vectorType,
                                      builder.getFloatAttr(elemType, 0)));
      auto cmp = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                               operands[0], zero);
      newOp = builder.create<arith::SelectOp>(
          loc, cmp, builder.create<math::FloorOp>(loc, resType, operands[0]),
          builder.create<math::CeilOp>(loc, resType, operands[0]));
    } else if (symbol == "__nv_sqrtf") {
      newOp = builder.create<math::SqrtOp>(loc, operands.front().getType(),
                                           operands);
    } else if (symbol == "__nv_rsqrtf") {
      newOp = builder.create<math::RsqrtOp>(loc, operands.front().getType(),
                                            operands);
    } else if (symbol == "__nv_isnanf") {
      // isnan(x) -> cmpf uno x, x, e.g.
      // cmpf uno 1.0, 1.0 -> false
      // cmpf uno nan, nan -> true
      auto resVectorType = VectorType::get(
          ArrayRef<int64_t>{vectorLength},
          dyn_cast<TensorType>(op.getResult().getType()).getElementType());
      auto cmpFOp = builder.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::UNO, operands.front(), operands.front());
      newOp = builder.create<arith::ExtSIOp>(loc, resVectorType, cmpFOp);
    } else if (symbol == "__nv_isinff") {
      auto elemType =
          dyn_cast<VectorType>(operands.front().getType()).getElementType();
      auto constPositiveInf =
          triton::gcu::createConstantInf(builder, loc, elemType);
      auto vectorType =
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elemType);
      auto broadCastPositiveInfOp = builder.create<vector::BroadcastOp>(
          loc, vectorType, constPositiveInf);
      auto positiveInput = builder.create<math::AbsFOp>(loc, operands.front());
      auto cmpFOpPositiveInf =
          builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                        positiveInput, broadCastPositiveInfOp);
      auto resVectorType = VectorType::get(
          ArrayRef<int64_t>{vectorLength},
          dyn_cast<TensorType>(op.getResult().getType()).getElementType());
      newOp =
          builder.create<arith::ExtSIOp>(loc, resVectorType, cmpFOpPositiveInf);
    } else if (symbol == "__nv_finitef") {
      // isfinitf(x) ->
      // %1 = fcmp uno %x, %x
      // %2 = absf %x
      // %3 = fcmp oeq %2, +inf
      // %4 = ori %1, %3
      // %5 = xori %4, true
      auto isNanfOp = builder.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::UNO, operands.front(), operands.front());
      auto elemType =
          dyn_cast<VectorType>(operands.front().getType()).getElementType();
      auto constPositiveInf =
          triton::gcu::createConstantInf(builder, loc, elemType);
      auto vectorType =
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elemType);
      auto broadCastPositiveInfOp = builder.create<vector::BroadcastOp>(
          loc, vectorType, constPositiveInf);
      auto positiveInput = builder.create<math::AbsFOp>(loc, operands.front());
      auto isInffOp =
          builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                        positiveInput, broadCastPositiveInfOp);
      auto oneMask =
          builder.create<arith::ConstantIntOp>(loc, 1, builder.getI32Type())
              .getResult();
      auto oneMaskVectorType = VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               builder.getI32Type());
      auto oneMaskVec =
          builder.create<vector::BroadcastOp>(loc, oneMaskVectorType, oneMask);
      auto isNanfOrInffOp =
          builder.create<arith::OrIOp>(loc, isNanfOp, isInffOp);
      auto resVectorType = VectorType::get(
          ArrayRef<int64_t>{vectorLength},
          dyn_cast<TensorType>(op.getResult().getType()).getElementType());
      auto isNanfOrInffOpI32 =
          builder.create<arith::ExtSIOp>(loc, resVectorType, isNanfOrInffOp);
      newOp = builder.create<arith::XOrIOp>(loc, isNanfOrInffOpI32, oneMaskVec);
    } else if (symbol == "__nv_rintf") {
      newOp = builder.create<math::RoundEvenOp>(loc, operands.front().getType(),
                                                operands);
    } else {
      llvm_unreachable(
          ("unsupported extern elementwise: " + symbol).str().c_str());
    }
    map.map(op.getResult(), newOp->getResult(0));
  }

  void handleCommonOp(Operation &op, OpBuilder &builder, IRMapping &map,
                      unsigned vectorLength) const {
    Operation *newOp;
    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      auto condition = selectOp.getCondition();
      auto mapValue = map.lookup(condition);
      if (cast<VectorType>(mapValue.getType()).getElementType().isInteger(8)) {
        map.map(condition,
                builder
                    .create<gcu::VectorConvertOp>(
                        op.getLoc(),
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        mapValue)
                    .getResult(0));
        newOp = builder.clone(op, map);
        map.map(condition, mapValue);
      } else {
        newOp = builder.clone(op, map);
      }
    } else if (auto cvtOp = dyn_cast<arith::ExtUIOp>(op)) {
      if (cast<TensorType>(cvtOp.getIn().getType())
              .getElementType()
              .isInteger(1) &&
          cast<TensorType>(cvtOp.getOut().getType())
              .getElementType()
              .isInteger(8)) {
        map.map(cvtOp.getOut(), map.lookup(cvtOp.getIn()));
        return;
      } else {
        newOp = builder.clone(op, map);
      }
    } else {
      newOp = builder.clone(op, map);
    }
    SmallVector<Type> resultTypes;
    auto typeInterface = dyn_cast<InferTypeOpInterface>(newOp);
    if (!typeInterface ||
        failed(typeInterface.inferReturnTypes(
            newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
            newOp->getAttrDictionary(), newOp->getPropertiesStorage(),
            newOp->getRegions(), resultTypes))) {
      resultTypes.clear();
      llvm::transform(
          op.getResultTypes(), std::back_inserter(resultTypes),
          [&](auto resultType) {
            return VectorType::get(
                ArrayRef<int64_t>{vectorLength},
                getTypeConverter()->convertType(
                    dyn_cast<TensorType>(resultType).getElementType()));
          });
    }

    for (auto [resultType, result, newResult] :
         llvm::zip(resultTypes, op.getResults(), newOp->getResults())) {
      newResult.setType(resultType);
      if (isa<arith::CmpFOp, arith::CmpIOp>(op)) {
        map.map(result, builder
                            .create<gcu::VectorConvertOp>(
                                op.getLoc(),
                                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                                builder.getIntegerType(8)),
                                newResult)
                            .getResult(0));
      } else {
        map.map(result, newResult);
      }
    }
  }
};
} // namespace

void mlir::triton::populateElementwiseFusionOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin) {
  patterns.add<GCUElementwiseFusionOpLowering>(converter, patterns.getContext(),
                                               userAnalysis, replaced2Origin);
}
