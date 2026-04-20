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

#include <functional>
#include <map>
#include <string>
#include <utility>

#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "PatternTritonGPUOpToGCU.h"

#include "Analysis/FirstLastUserAnalysis.h"
#include "TritonGCUToGCU/TritonGCUToGCUUtils.h"
#include "Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {
SmallVector<Value> vectorizeCombineOpWithoutTerminator(
    Location loc, OpBuilder &builder, Region &combineOp, ValueRange operands,
    unsigned vectorLength, bool needCvtDataLayout = false) {
  IRMapping map;
  for (auto [arg, operand] : llvm::zip(combineOp.getArguments(), operands)) {
    map.map(arg, operand);
  }
  for (auto &o : combineOp.back().without_terminator()) {
    for (auto operand : o.getOperands()) {
      if (auto constantOp = operand.getDefiningOp<arith::ConstantOp>()) {
        if (!map.lookupOrNull(operand)) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(constantOp);
          if (operand.getType().isInteger(1)) {
            auto boolAttr = dyn_cast<BoolAttr>(constantOp.getValue());
            auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
            if ((boolAttr && !boolAttr.getValue()) ||
                (integerAttr && integerAttr.getValue().isZero())) {
              map.map(operand,
                      builder.create<vector::ConstantMaskOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          operand.getType()),
                          DenseI64ArrayAttr::get(builder.getContext(),
                                                 ArrayRef<int64_t>{0})));
            } else {
              map.map(
                  operand,
                  builder.create<vector::ConstantMaskOp>(
                      loc,
                      VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      operand.getType()),
                      DenseI64ArrayAttr::get(builder.getContext(),
                                             ArrayRef<int64_t>{vectorLength})));
            }
          } else {
            map.map(operand,
                    builder.create<vector::BroadcastOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        operand.getType()),
                        operand));
          }
        }
      }
    }
    Operation *newOp;
    if (auto selectOp = dyn_cast<arith::SelectOp>(o)) {
      auto condition = selectOp.getCondition();
      auto mapValue = map.lookup(condition);
      if (cast<VectorType>(mapValue.getType()).getElementType().isInteger(8)) {
        map.map(condition,
                builder
                    .create<gcu::VectorConvertOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        mapValue)
                    .getResult(0));
        newOp = builder.clone(o, map);
        map.map(condition, mapValue);
      } else {
        newOp = builder.clone(o, map);
      }
    } else {
      newOp = builder.clone(o, map);
    }
    SmallVector<Type> resultTypes;
    auto typeInterface = dyn_cast<InferTypeOpInterface>(newOp);
    if (typeInterface &&
        succeeded(typeInterface.inferReturnTypes(
            newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
            newOp->getAttrDictionary(), newOp->getPropertiesStorage(),
            newOp->getRegions(), resultTypes))) {
      for (auto [resultType, result, newResult] :
           llvm::zip(resultTypes, o.getResults(), newOp->getResults())) {
        newResult.setType(resultType);
        map.map(result, newResult);
      }
    } else {
      for (auto [result, newResult] :
           llvm::zip(o.getResults(), newOp->getResults())) {
        auto vectorTy =
            VectorType::get(ArrayRef<int64_t>{vectorLength}, result.getType());
        newResult.setType(vectorTy);
        map.map(result, newResult);
      }
    }
  }
  auto terminatorOprands = llvm::to_vector(llvm::map_range(
      llvm::cast<triton::ScanReturnOp>(combineOp.back().getTerminator())
          .getResult(),
      [&](auto v) {
        auto mappingValue = map.lookupOrNull(v);
        assert(mappingValue != nullptr);
        if (v.getType().isInteger(1) && needCvtDataLayout) {
          mappingValue =
              builder
                  .create<gcu::VectorConvertOp>(
                      loc,
                      VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      builder.getIntegerType(8)),
                      mappingValue)
                  .getResult(0);
        }
        return mappingValue;
      }));
  return terminatorOprands;
}

void vectorizeCombineOpTerminator(Location loc, OpBuilder &builder,
                                  ValueRange operands) {
  builder.create<triton::ScanReturnOp>(loc, operands);
}

struct TTScanOpLowering : SharedConversionPattern<triton::ScanOp> {
  using SharedConversionPattern::SharedConversionPattern;
  void applyScan(triton::ScanOp op, OpBuilder &rewriter,
                 ArrayRef<Value> outputs, ArrayRef<Value> inputs, Type type,
                 unsigned vectorLength, bool reverse) const {
    auto axis = op.getAxis();
    auto loc = op.getLoc();
    auto numElems = triton::gcu::getElemsPerThread(type);
    auto numOutput = outputs.size();
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(type);
    auto tag = pTagPool.getSyncTagInfo(op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // initialize outputs by inputs
    for (unsigned i = 0; i < numOutput; ++i) {
      rewriter.create<memref::DmaStartOp>(
          loc, inputs[i], SmallVector<Value, 4>(numElems.size(), zero),
          outputs[i], SmallVector<Value, 4>(numElems.size(), zero),
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems),
          tag.getTag(), ValueRange{tag.getIdx()});
      rewriter.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
    }

    std::array<int64_t, 3> scanInOutDims = {1, 1, 1};
    int64_t scanAxis = 2;
    for (int i = numElems.size() - 1, j = 2; i >= 0; i--) {
      if (static_cast<unsigned>(i) == axis) {
        if (scanInOutDims[j] == 1) {
          scanInOutDims[j] = numElems[i];
        } else {
          scanInOutDims[--j] = numElems[i];
        }
        scanAxis = j;
        --j;
      } else {
        scanInOutDims[j] *= numElems[i];
      }
    }
    SmallVector<Value, 4> outs;
    llvm::transform(outputs, std::back_inserter(outs), [&](auto output) {
      return rewriter.create<memref::ReinterpretCastOp>(
          loc,
          MemRefType::get(scanInOutDims,
                          cast<MemRefType>(output.getType()).getElementType()),
          output, ValueRange{}, ValueRange{}, ValueRange{},
          ArrayRef<int64_t>{0},
          ArrayRef<int64_t>{scanInOutDims[0], scanInOutDims[1],
                            scanInOutDims[2]},
          ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                            scanInOutDims[2], 1});
    });
    if (succeeded(applyGeneralScan(op, rewriter, outs, scanInOutDims, scanAxis,
                                   vectorLength, reverse))) {
      return;
    }
    return applyScalarImpl(op, rewriter, outs, scanInOutDims, scanAxis,
                           reverse);
  }

  LogicalResult applyGeneralScan(triton::ScanOp op, OpBuilder &rewriter,
                                 ArrayRef<Value> outputs,
                                 const std::array<int64_t, 3> &scanInOutDims,
                                 int64_t scanAxis, unsigned vectorLength,
                                 bool reverse) const {
    auto loc = op.getLoc();
    int64_t vectorizeAxis;
    if (scanAxis == 2) {
      assert(scanInOutDims[0] == 1);
      vectorizeAxis = 1;
    } else {
      assert(scanAxis == 1);
      vectorizeAxis = scanInOutDims[0] > scanInOutDims[2] ? 0 : 2;
    }
    unsigned bpe = 4; // gatherscatter offset, i32
    for (auto output : outputs) {
      auto elementTy = cast<MemRefType>(output.getType()).getElementType();
      auto bytes = triton::gcu::getBpe(elementTy);
      bpe = bytes > bpe ? bytes : bpe;
    }
    if (scanInOutDims[vectorizeAxis] < vectorLength) {
      return failure();
    }

    unsigned maxBpe = 1;
    unsigned minBpe = 4;
    auto target_supporti64 =
        !triton::gcu::get_bool_env("ENABLE_I64_CHECK", true);
    bool is_i64 = false;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (elementType.isInteger(64) && target_supporti64)
        is_i64 = true;
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32() &&
          (target_supporti64 && !elementType.isInteger(64))) {
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    if (is_i64)
      minBpe = 8;
    auto numVacc = maxBpe / minBpe;
    if (numVacc > 4) {
      return failure();
    }
    while (numVacc < targetInfo[arch].preferVaccNum) {
      unsigned vLen = 2 * vectorLength;
      if (scanInOutDims[vectorizeAxis] < vLen) {
        break;
      }
      numVacc *= 2;
      vectorLength = vLen;
    }
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<VectorType, 4> vectorTypes;
    llvm::transform(
        outputs, std::back_inserter(vectorTypes), [vectorLength](auto output) {
          auto elementTy = cast<MemRefType>(output.getType()).getElementType();
          return VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
        });

    SmallVector<Value, 4> lbs(scanInOutDims.size(), zero);
    lbs[scanAxis] = one;
    std::array<int64_t, 3> loopcnt = scanInOutDims;
    if (loopcnt[vectorizeAxis] % vectorLength != 0) {
      llvm_unreachable("invalid datalayout");
    }
    loopcnt[vectorizeAxis] /= vectorLength;
    SmallVector<Value, 4> ubs{
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[1]),
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[2])};
    SmallVector<Value, 4> step(scanInOutDims.size(), one);

    auto maskType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI1Type());
    Value mask = rewriter.create<vector::ConstantMaskOp>(
        loc, maskType,
        DenseI64ArrayAttr::get(rewriter.getContext(),
                               ArrayRef<int64_t>{vectorLength}));
    unsigned strideOnVectorizeAxis =
        std::accumulate(scanInOutDims.begin() + vectorizeAxis + 1,
                        scanInOutDims.end(), 1, std::multiplies<unsigned>());
    auto vecTy =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI32Type());
    auto indexVec = rewriter.create<arith::MulIOp>(
        loc,
        rewriter
            .create<gcu::VectorConvertOp>(
                loc, vecTy,
                rewriter
                    .create<vector::StepOp>(
                        loc, VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             rewriter.getIndexType()))
                    .getResult())
            .getResult(0),
        rewriter.create<vector::BroadcastOp>(
            loc, vecTy,
            rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32Type(),
                rewriter.getI32IntegerAttr(strideOnVectorizeAxis))));

    SmallVector<Value, 4> passThruValues;
    for (unsigned i = 0; i < numOutput; ++i) {
      passThruValues.push_back(rewriter.create<vector::BroadcastOp>(
          loc, vectorTypes[i],
          rewriter.create<arith::ConstantOp>(
              loc, vectorTypes[i].getElementType(),
              rewriter.getZeroAttr(vectorTypes[i].getElementType()))));
    }

    scf::buildLoopNest(
        rewriter, loc,
        ArrayRef<Value>(lbs.begin(), lbs.begin() + vectorizeAxis),
        ArrayRef<Value>(ubs.begin(), ubs.begin() + vectorizeAxis),
        ArrayRef<Value>(step.begin(), step.begin() + vectorizeAxis),
        [&](OpBuilder &builder, Location loc, ValueRange outerIters) {
          scf::buildLoopNest(
              rewriter, loc,
              ArrayRef<Value>(lbs.begin() + vectorizeAxis, lbs.end()),
              ArrayRef<Value>(ubs.begin() + vectorizeAxis, ubs.end()),
              ArrayRef<Value>(step.begin() + vectorizeAxis, step.end()),
              [&](OpBuilder &builder, Location loc, ValueRange innerIters) {
                SmallVector<Value, 4> inputIndices;
                SmallVector<Value, 4> outputIndices;

                SmallVector<Type, 4> resultElemTypes;
                SmallVector<Value, 4> operands;
                SmallVector<Value, 4> ivs;
                for (auto iv : outerIters) {
                  ivs.push_back(iv);
                }
                for (auto iv : innerIters) {
                  ivs.push_back(iv);
                }
                if (reverse) {
                  ivs[scanAxis] = builder.create<arith::SubIOp>(
                      loc,
                      builder.create<arith::ConstantIndexOp>(
                          loc, scanInOutDims[scanAxis] - 1),
                      ivs[scanAxis]);
                }
                for (unsigned i = 0; i < ivs.size(); ++i) {
                  if (i == vectorizeAxis) {
                    outputIndices.push_back(builder.create<arith::MulIOp>(
                        loc, ivs[i],
                        rewriter.create<arith::ConstantIndexOp>(loc,
                                                                vectorLength)));
                  } else {
                    outputIndices.push_back(ivs[i]);
                  }
                  if (i == scanAxis) {
                    if (reverse) {
                      inputIndices.push_back(builder.create<arith::AddIOp>(
                          loc, outputIndices[i], one));
                    } else {
                      inputIndices.push_back(builder.create<arith::SubIOp>(
                          loc, outputIndices[i], one));
                    }
                  } else {
                    inputIndices.push_back(outputIndices[i]);
                  }
                }

                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], outputs[i], inputIndices, indexVec,
                      mask, passThruValues[i]));
                }
                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], outputs[i], outputIndices, indexVec,
                      mask, passThruValues[i]));
                  resultElemTypes.push_back(vectorTypes[i]);
                }

                auto executeRegionOp =
                    builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
                executeRegionOp.getRegion().emplaceBlock();
                IRMapping map;
                for (auto [arg, operand] :
                     llvm::zip(op.getCombineOp().getArguments(), operands)) {
                  map.map(arg, operand);
                }
                {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.setInsertionPointToStart(
                      &executeRegionOp.getRegion().back());
                  for (auto &o : op.getCombineOp().back()) {
                    for (auto operand : o.getOperands()) {
                      if (auto constantOp =
                              operand.getDefiningOp<arith::ConstantOp>()) {
                        if (!map.lookupOrNull(operand)) {
                          OpBuilder::InsertionGuard guard(builder);
                          builder.setInsertionPointAfter(constantOp);
                          map.map(operand,
                                  builder.create<vector::BroadcastOp>(
                                      loc,
                                      VectorType::get(
                                          ArrayRef<int64_t>{vectorLength},
                                          operand.getType()),
                                      operand));
                        }
                      }
                    }
                    auto newO = builder.clone(o, map);
                    for (auto [result, newResult] :
                         llvm::zip(o.getResults(), newO->getResults())) {
                      auto vectorTy = VectorType::get(
                          ArrayRef<int64_t>{vectorLength}, result.getType());
                      newResult.setType(vectorTy);
                      map.map(result, newResult);
                    }
                  }
                }

                for (unsigned i = 0; i < numOutput; ++i) {
                  builder.create<vector::ScatterOp>(
                      loc, outputs[i], outputIndices, indexVec, mask,
                      executeRegionOp.getResult(i));
                }
              });
        });
    doMemFence(rewriter, op);
    return success();
  }

  void applyScalarImpl(triton::ScanOp op, OpBuilder &rewriter,
                       ArrayRef<Value> outputs,
                       const std::array<int64_t, 3> &scanInOutDims,
                       int64_t scanAxis, bool reverse) const {
    auto loc = op.getLoc();
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value, 4> lbs(scanInOutDims.size(), zero);
    lbs[scanAxis] = one;
    SmallVector<Value, 4> ubs{
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[1]),
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[2])};

    scf::buildLoopNest(
        rewriter, loc, lbs, ubs,
        SmallVector<Value, 4>(scanInOutDims.size(), one),
        [&](OpBuilder &builder, Location loc, ValueRange iters) {
          SmallVector<Value, 4> outputIters(iters.begin(), iters.end());
          if (reverse) {
            outputIters[scanAxis] = builder.create<arith::SubIOp>(
                loc,
                builder.create<arith::ConstantIndexOp>(
                    loc, scanInOutDims[scanAxis] - 1),
                outputIters[scanAxis]);
          }

          SmallVector<Value, 4> operands;
          SmallVector<Type, 4> resultElemTypes;
          SmallVector<Value, 4> inputIters(outputIters.begin(),
                                           outputIters.end());
          if (reverse) {
            inputIters[scanAxis] =
                builder.create<arith::AddIOp>(loc, one, inputIters[scanAxis]);
          } else {
            inputIters[scanAxis] =
                builder.create<arith::SubIOp>(loc, inputIters[scanAxis], one);
          }

          for (unsigned i = 0; i < numOutput; ++i) {
            operands.push_back(
                builder.create<memref::LoadOp>(loc, outputs[i], inputIters));
          }
          for (unsigned i = 0; i < numOutput; ++i) {
            operands.push_back(
                builder.create<memref::LoadOp>(loc, outputs[i], outputIters));
            resultElemTypes.push_back(operands.back().getType());
          }

          auto executeRegion =
              builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
          executeRegion.getRegion().emplaceBlock();
          IRMapping map;
          for (auto [arg, operand] :
               llvm::zip(op.getCombineOp().getArguments(), operands)) {
            map.map(arg, operand);
          }
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&executeRegion.getRegion().back());
            for (auto &o : op.getCombineOp().back()) {
              auto newO = builder.clone(o, map);
              for (auto [result, newResult] :
                   llvm::zip(o.getResults(), newO->getResults())) {
                map.map(result, newResult);
              }
            }
          }

          for (unsigned i = 0; i < numOutput; ++i) {
            builder.create<memref::StoreOp>(loc, executeRegion.getResult(i),
                                            outputs[i], outputIters);
          }
        });

    doMemFence(rewriter, op);
  }

  void applyVectorizationImpl(triton::ScanOp op, OpBuilder &rewriter,
                              ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                              const std::array<int64_t, 3> &scanInOutDims,
                              int64_t scanAxis, unsigned vectorLength,
                              bool reverse) const {
    auto loc = op.getLoc();
    auto numOutput = outputs.size();
    assert(scanAxis == 1);
    auto loopLimit = scanInOutDims[1];
    auto vLength = rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
    constexpr int loopUnrollTime = 16;
    auto loopCnt = loopLimit > loopUnrollTime ? loopUnrollTime : loopLimit;
    auto loopCntValue = rewriter.create<arith::ConstantIndexOp>(loc, loopCnt);
    SmallVector<VectorType, 4> vectorTypes;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      vectorTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elementType));
    }
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto &combineOp = op.getCombineOp();
    rewriter.create<scf::ForOp>(
        loc, zero,
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[0]), one,
        ValueRange{},
        [&](OpBuilder &builder, Location loc, Value iter0,
            ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zero,
              builder.create<arith::ConstantIndexOp>(loc, scanInOutDims[2]),
              vLength, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iter2,
                  ValueRange iterArgs) {
                SmallVector<Value, loopUnrollTime> vectorList;
                for (unsigned i = 0; i < loopCnt; ++i) {
                  Value scanIndex;
                  if (!reverse) {
                    scanIndex = builder.create<arith::ConstantIndexOp>(loc, i);
                  } else {
                    scanIndex = builder.create<arith::ConstantIndexOp>(
                        loc, scanInOutDims[1] - 1 - i);
                  }
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList.emplace_back(builder.create<vector::LoadOp>(
                        loc, vectorTypes[j], inputs[j],
                        ValueRange{iter0, scanIndex, iter2}));
                  }
                }
                // scan 8 group
                SmallVector<Value, 4> args(numOutput * 2);
                for (unsigned i = 0; i < loopCnt / 2; ++i) {
                  for (unsigned j = 0; j < numOutput; ++j) {
                    args[j] = vectorList[(i * 2) * numOutput + j];
                    args[numOutput + j] =
                        vectorList[(i * 2 + 1) * numOutput + j];
                  }
                  auto combinOut = vectorizeCombineOpWithoutTerminator(
                      loc, builder, combineOp, args, vectorLength);
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList[(i * 2 + 1) * numOutput + j] = combinOut[j];
                  }
                }
                // group 2 result
                for (unsigned i = 1; i < loopCnt / 2; ++i) {
                  for (unsigned j = 0; j < numOutput; ++j) {
                    args[j] = vectorList[((i - 1) * 2 + 1) * numOutput + j];
                    args[numOutput + j] =
                        vectorList[(i * 2 + 1) * numOutput + j];
                  }
                  auto combinOut = vectorizeCombineOpWithoutTerminator(
                      loc, builder, combineOp, args, vectorLength);
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList[(i * 2 + 1) * numOutput + j] = combinOut[j];
                  }
                }
                // update first group
                for (unsigned i = 1; i < loopCnt / 2; ++i) {
                  for (unsigned j = 0; j < numOutput; ++j) {
                    args[j] = vectorList[((i - 1) * 2 + 1) * numOutput + j];
                    args[numOutput + j] = vectorList[(i * 2) * numOutput + j];
                  }
                  auto combinOut = vectorizeCombineOpWithoutTerminator(
                      loc, builder, combineOp, args, vectorLength);
                  for (unsigned j = 0; j < numOutput; ++j) {
                    vectorList[(i * 2) * numOutput + j] = combinOut[j];
                  }
                }
                for (unsigned i = 0; i < loopCnt; ++i) {
                  Value scanIndex;
                  if (!reverse) {
                    scanIndex = builder.create<arith::ConstantIndexOp>(loc, i);
                  } else {
                    scanIndex = builder.create<arith::ConstantIndexOp>(
                        loc, scanInOutDims[1] - 1 - i);
                  }
                  for (unsigned j = 0; j < numOutput; ++j) {
                    builder.create<vector::StoreOp>(
                        loc, vectorList[i * numOutput + j], outputs[j],
                        ValueRange{iter0, scanIndex, iter2});
                  }
                }
                SmallVector<Value, 4> blockOut(numOutput);
                for (unsigned j = 0; j < numOutput; ++j) {
                  blockOut[j] = vectorList[(loopCnt - 1) * numOutput + j];
                }
                builder.create<scf::ForOp>(
                    loc, loopCntValue,
                    builder.create<arith::ConstantIndexOp>(loc,
                                                           scanInOutDims[1]),
                    loopCntValue, blockOut,
                    [&](OpBuilder &builder, Location loc, Value iter1,
                        ValueRange lastBlockOut) {
                      SmallVector<Value, loopUnrollTime> vectorListInner;
                      for (unsigned i = 0; i < loopCnt; ++i) {
                        Value scanIndex = builder.create<arith::AddIOp>(
                            loc, builder.create<arith::ConstantIndexOp>(loc, i),
                            iter1);
                        if (reverse) {
                          scanIndex = builder.create<arith::SubIOp>(
                              loc,
                              builder.create<arith::ConstantIndexOp>(
                                  loc, scanInOutDims[1] - 1),
                              scanIndex);
                        }
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner.emplace_back(
                              builder.create<vector::LoadOp>(
                                  loc, vectorTypes[j], inputs[j],
                                  ValueRange{iter0, scanIndex, iter2}));
                        }
                      }
                      // scan 8 group
                      SmallVector<Value, 4> args(numOutput * 2);
                      for (unsigned i = 0; i < loopCnt / 2; ++i) {
                        for (unsigned j = 0; j < numOutput; ++j) {
                          args[j] = vectorListInner[(i * 2) * numOutput + j];
                          args[numOutput + j] =
                              vectorListInner[(i * 2 + 1) * numOutput + j];
                        }
                        auto combinOut = vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp, args, vectorLength);
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner[(i * 2 + 1) * numOutput + j] =
                              combinOut[j];
                        }
                      }
                      // update group  result
                      for (unsigned j = 0; j < numOutput; ++j) {
                        args[j] = lastBlockOut[j];
                        args[numOutput + j] =
                            vectorListInner[(1) * numOutput + j];
                      }
                      auto combinOut = vectorizeCombineOpWithoutTerminator(
                          loc, builder, combineOp, args, vectorLength);
                      for (unsigned j = 0; j < numOutput; ++j) {
                        vectorListInner[(1) * numOutput + j] = combinOut[j];
                      }

                      for (unsigned i = 1; i < loopCnt / 2; ++i) {
                        for (unsigned j = 0; j < numOutput; ++j) {
                          args[j] =
                              vectorListInner[((i - 1) * 2 + 1) * numOutput +
                                              j];
                          args[numOutput + j] =
                              vectorListInner[(i * 2 + 1) * numOutput + j];
                        }
                        auto combinOut = vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp, args, vectorLength);
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner[(i * 2 + 1) * numOutput + j] =
                              combinOut[j];
                        }
                      }
                      // update first group
                      for (unsigned j = 0; j < numOutput; ++j) {
                        args[j] = lastBlockOut[j];
                        args[numOutput + j] = vectorListInner[j];
                      }
                      combinOut = vectorizeCombineOpWithoutTerminator(
                          loc, builder, combineOp, args, vectorLength);
                      for (unsigned j = 0; j < numOutput; ++j) {
                        vectorListInner[j] = combinOut[j];
                      }
                      for (unsigned i = 1; i < loopCnt / 2; ++i) {
                        for (unsigned j = 0; j < numOutput; ++j) {
                          args[j] =
                              vectorListInner[((i - 1) * 2 + 1) * numOutput +
                                              j];
                          args[numOutput + j] =
                              vectorListInner[(i * 2) * numOutput + j];
                        }
                        auto combinOut = vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp, args, vectorLength);
                        for (unsigned j = 0; j < numOutput; ++j) {
                          vectorListInner[(i * 2) * numOutput + j] =
                              combinOut[j];
                        }
                      }
                      for (unsigned i = 0; i < loopCnt; ++i) {
                        Value scanIndex = builder.create<arith::AddIOp>(
                            loc, builder.create<arith::ConstantIndexOp>(loc, i),
                            iter1);
                        if (reverse) {
                          scanIndex = builder.create<arith::SubIOp>(
                              loc,
                              builder.create<arith::ConstantIndexOp>(
                                  loc, scanInOutDims[1] - 1),
                              scanIndex);
                        }
                        for (unsigned j = 0; j < numOutput; ++j) {
                          builder.create<vector::StoreOp>(
                              loc, vectorListInner[i * numOutput + j],
                              outputs[j], ValueRange{iter0, scanIndex, iter2});
                        }
                      }
                      SmallVector<Value, 4> blockOut(numOutput);
                      for (unsigned j = 0; j < numOutput; ++j) {
                        blockOut[j] =
                            vectorListInner[(loopCnt - 1) * numOutput + j];
                      }
                      vectorizeCombineOpTerminator(loc, builder, blockOut);
                    });
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
  }

  LogicalResult applyVectorizationScan(triton::ScanOp op, OpBuilder &rewriter,
                                       ArrayRef<Value> outputs,
                                       ArrayRef<Value> inputs,
                                       std::array<int64_t, 3> scanInOutDims,
                                       int64_t scanAxis, unsigned vectorLength,
                                       bool reverse) const {
    auto loc = op.getLoc();
    auto numOutput = op.getResults().size();
    unsigned maxBpe = 1;
    unsigned minBpe = 4;
    auto target_supporti64 =
        !triton::gcu::get_bool_env("ENABLE_I64_CHECK", true);
    bool is_i64 = false;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (elementType.isInteger(64) && target_supporti64)
        is_i64 = true;
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32() &&
          (target_supporti64 && !elementType.isInteger(64))) {
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    if (is_i64)
      minBpe = 8;
    auto numVacc = maxBpe / minBpe;
    if (numVacc > 4) {
      return failure();
    }
    if (scanAxis == 1 && scanInOutDims[2] >= vectorLength &&
        scanInOutDims[1] >= 8) {
      auto numVacc = maxBpe / minBpe;
      while (numVacc < targetInfo[arch].preferVaccNum) {
        unsigned vLen = 2 * vectorLength;
        if (scanInOutDims[2] < vLen) {
          break;
        }
        numVacc *= 2;
        vectorLength = vLen;
      }
      SmallVector<Value, 4> sanInputs;
      SmallVector<Value, 4> sanOutputs;
      llvm::transform(inputs, std::back_inserter(sanInputs), [&](auto input) {
        return rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(scanInOutDims,
                            cast<MemRefType>(input.getType()).getElementType()),
            input, ValueRange{}, ValueRange{}, ValueRange{},
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
            ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                              scanInOutDims[2], 1});
      });
      llvm::transform(
          outputs, std::back_inserter(sanOutputs), [&](auto output) {
            return rewriter.create<memref::ReinterpretCastOp>(
                loc,
                MemRefType::get(
                    scanInOutDims,
                    cast<MemRefType>(output.getType()).getElementType()),
                output, ValueRange{}, ValueRange{}, ValueRange{},
                ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
                ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                                  scanInOutDims[2], 1});
          });
      applyVectorizationImpl(op, rewriter, sanOutputs, sanInputs, scanInOutDims,
                             scanAxis, vectorLength, op.getReverse());
      return success();
    } else if (scanAxis == 2 && scanInOutDims[1] >= vectorLength &&
               scanInOutDims[2] >= 8) {
      // need transpose
      auto numVacc = maxBpe / minBpe;
      while (numVacc < targetInfo[arch].preferVaccNum) {
        unsigned vLen = 2 * vectorLength;
        if (scanInOutDims[1] < vLen) {
          break;
        }
        numVacc *= 2;
        vectorLength = vLen;
      }
      SmallVector<Value, 4> sanInputs;
      SmallVector<Value, 4> sanOutputs;
      llvm::transform(inputs, std::back_inserter(sanInputs), [&](auto input) {
        return rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(scanInOutDims,
                            cast<MemRefType>(input.getType()).getElementType()),
            input, ValueRange{}, ValueRange{}, ValueRange{},
            ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
            ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                              scanInOutDims[2], 1});
      });
      llvm::transform(
          outputs, std::back_inserter(sanOutputs), [&](auto output) {
            return rewriter.create<memref::ReinterpretCastOp>(
                loc,
                MemRefType::get(
                    scanInOutDims,
                    cast<MemRefType>(output.getType()).getElementType()),
                output, ValueRange{}, ValueRange{}, ValueRange{},
                ArrayRef<int64_t>{0}, ArrayRef<int64_t>{scanInOutDims},
                ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                                  scanInOutDims[2], 1});
          });
      std::array<int64_t, 3> transposeLayout = {0, 1, 2};
      transposeLayout[2] = 1;
      transposeLayout[1] = 2;
      SmallVector<Type> elementTypes;
      for (auto output : outputs) {
        auto elementType = cast<MemRefType>(output.getType()).getElementType();
        elementTypes.push_back(elementType);
      }
      scanAxis = transposeLayout[scanAxis];
      int64_t dim0 = scanInOutDims[transposeLayout[0]];
      int64_t dim1 = scanInOutDims[transposeLayout[1]];
      int64_t dim2 = scanInOutDims[transposeLayout[2]];
      scanInOutDims[0] = dim0;
      scanInOutDims[1] = dim1;
      scanInOutDims[2] = dim2;
      SmallVector<Value, 3> transposeLayoutValue;
      SmallVector<Value, 2> tmpBuffers;
      llvm::transform(transposeLayout, std::back_inserter(transposeLayoutValue),
                      [&](auto dim) {
                        return rewriter.create<arith::ConstantIntOp>(
                            loc, rewriter.getI32Type(), dim);
                      });
      auto tag = pTagPool.getSyncTagInfo(op);
      llvm::transform(
          sanInputs, std::back_inserter(tmpBuffers), [&](auto input) {
            auto memrefTy = cast<MemRefType>(input.getType());
            auto elementTy = memrefTy.getElementType();
            auto tmpBuffer = rewriter.create<memref::AllocOp>(
                loc,
                MemRefType::get(ArrayRef<int64_t>{scanInOutDims}, elementTy));
            rewriter.create<memref_ext::TransposeStartOp>(
                loc, tmpBuffer, input, transposeLayoutValue, tag.getTag(),
                ValueRange{tag.getIdx()});
            rewriter.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()},
                rewriter.create<arith::ConstantIndexOp>(
                    loc, memrefTy.getNumElements()));
            return tmpBuffer;
          });
      SmallVector<Value, 2> sanTransOutputs(tmpBuffers.begin(),
                                            tmpBuffers.end());
      sanInputs = SmallVector<Value, 4>(tmpBuffers.begin(), tmpBuffers.end());
      // do scan with sanTransOutputs  scanAxis==1
      applyVectorizationImpl(op, rewriter, sanTransOutputs, sanInputs,
                             scanInOutDims, scanAxis, vectorLength,
                             op.getReverse());
      //  transpose back
      SmallVector<Value, 2> tmpoutBuffers;
      std::array<int64_t, 3> transposeOutLayout = {0, 1, 2};
      transposeOutLayout[2] = 1;
      transposeOutLayout[1] = 2;
      dim0 = scanInOutDims[transposeOutLayout[0]];
      dim1 = scanInOutDims[transposeOutLayout[1]];
      dim2 = scanInOutDims[transposeOutLayout[2]];
      scanInOutDims[0] = dim0;
      scanInOutDims[1] = dim1;
      scanInOutDims[2] = dim2;
      SmallVector<Value, 3> transposeOutLayoutValue;
      llvm::transform(transposeOutLayout,
                      std::back_inserter(transposeOutLayoutValue),
                      [&](auto dim) {
                        return rewriter.create<arith::ConstantIntOp>(
                            loc, rewriter.getI32Type(), dim);
                      });
      for (unsigned i = 0; i < numOutput; ++i) {
        rewriter.create<memref_ext::TransposeStartOp>(
            loc, sanOutputs[i], sanTransOutputs[i], transposeOutLayoutValue,
            tag.getTag(), ValueRange{tag.getIdx()});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag.getTag(), ValueRange{tag.getIdx()},
            rewriter.create<arith::ConstantIndexOp>(
                loc, scanInOutDims[0] * scanInOutDims[1] * scanInOutDims[2]));
      }
      for (auto buffer : tmpBuffers) {
        rewriter.create<memref::DeallocOp>(loc, buffer);
      }
      return success();
    }
    return failure();
  }

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto inputType = dyn_cast<TensorType>(op.getSrcs()[0].getType());
    auto numElems = triton::gcu::getElemsPerThread(inputType);
    auto slicedAxies = getSlicedAxies(inputType);
    auto axis = op.getAxis();
    bool isScanDimSplit = slicedAxies.count(axis);
    auto reverse = op.getReverse();
    auto numInput = op.getSrcs().size();
    auto numOutput = op.getResults().size();

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // create outputs
    SmallVector<Value, 4> outputs;
    SmallVector<Type, 4> outputElemTypes;
    SmallVector<std::pair<Operation *, int>, 4> lastUsers;
    unsigned maxBpe = 1;
    unsigned minBpe = 4;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto resultType =
          dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType(i)));
      auto elemType = resultType.getElementType();
      outputElemTypes.push_back(elemType);
      auto lastUser = userAnalysis.getLastUser(op.getResults()[i]);
      Value output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                 replaced2Origin, resultType);
      outputs.push_back(output);
      lastUsers.push_back(lastUser);
      auto bpe = mlir::triton::gcu::getBpe(elemType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    unsigned vectorLength = targetInfo[arch].vaccSizeInBytes / minBpe;
    std::array<int64_t, 3> scanInOutDims = {1, 1, 1};
    int64_t scanAxis = 2;
    for (int i = numElems.size() - 1, j = 2; i >= 0; i--) {
      if (static_cast<unsigned>(i) == axis) {
        if (scanInOutDims[j] == 1) {
          scanInOutDims[j] = numElems[i];
        } else {
          scanInOutDims[--j] = numElems[i];
        }
        scanAxis = j;
        --j;
      } else {
        scanInOutDims[j] *= numElems[i];
      }
    }
    assert(scanAxis == 1 || scanAxis == 2);
    auto encodingAttr = dyn_cast<RankedTensorType>(inputType).getEncoding();
    auto warpsPerCTA = triton::gcu::getWarpsPerCTA(encodingAttr);
    auto threadsPerWarp =
        triton::gpu::getThreadsPerWarp(dyn_cast<RankedTensorType>(inputType));
    auto elementsPerThread = triton::gcu::getElemsPerThread(inputType);
    bool isValidBlockEncoding = true;
    for (auto [dim, elems, threads, warps] :
         llvm::zip(inputType.getShape(), elementsPerThread, threadsPerWarp,
                   warpsPerCTA)) {
      if (dim != elems * threads * warps) {
        isValidBlockEncoding = false;
        break;
      }
    }
    bool shouldonlyMasterSip = false;
    if ((!slicedAxies.empty()) && scanAxis == 2 &&
        (scanInOutDims[1] < vectorLength || scanInOutDims[2] < 8)) {
      shouldonlyMasterSip = true;
    } else if ((!slicedAxies.empty()) && scanAxis == 1 &&
               (scanInOutDims[2] < vectorLength || scanInOutDims[1] < 8)) {
      shouldonlyMasterSip = true;
    }
    if (shouldonlyMasterSip || !isValidBlockEncoding || isScanDimSplit) {
      auto tag = pTagPool.getSyncTagInfo(op);
      // move to shared memory
      SmallVector<Value, 4> sharedInputs;
      for (unsigned i = 0; i < numInput; ++i) {
        sharedInputs.push_back(storeToSharedMem(
            rewriter, tag,
            dyn_cast<RankedTensorType>(op.getSrcs()[i].getType()),
            adaptor.getSrcs()[i], false, std::make_pair(op.getOperation(), -1),
            userAnalysis, replaced2Origin));
      }

      // load all shared memory to thread 0
      SmallVector<Value, 4> mergedInputs;
      RankedTensorType mergedInputType;
      for (unsigned i = 0; i < numInput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getSrcs()[i].getType());
        auto tensorType =
            RankedTensorType::get(tType.getShape(), tType.getElementType(),
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        mergedInputType = tensorType;
        mergedInputs.push_back(loadFromSharedMem(
            rewriter, tag, tensorType, sharedInputs[i], true,
            std::make_pair(op.getOperation(), -1), std::make_pair(nullptr, -1),
            userAnalysis, replaced2Origin));
      }

      SmallVector<Value, 4> mergedOutputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getResultTypes()[i]);
        auto tensorType =
            RankedTensorType::get(tType.getShape(), tType.getElementType(),
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        auto resultType =
            dyn_cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        mergedOutputs.push_back(
            syncAllocOp(rewriter, loc, std::make_pair(op.getOperation(), -1),
                        userAnalysis, replaced2Origin, resultType));
      }

      // computing in     0
      auto isThread0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
      rewriter.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            std::array<int64_t, 3> mergeInOutDims = {1, 1, 1};
            int64_t mergeScanAxis = 2;
            auto mergeShape = mergedInputType.getShape();
            for (int i = mergeShape.size() - 1, j = 2; i >= 0; i--) {
              if (static_cast<unsigned>(i) == axis) {
                if (mergeInOutDims[j] == 1) {
                  mergeInOutDims[j] = mergeShape[i];
                } else {
                  mergeInOutDims[--j] = mergeShape[i];
                }
                mergeScanAxis = j;
                --j;
              } else {
                mergeInOutDims[j] *= mergeShape[i];
              }
            }
            if (!succeeded(applyVectorizationScan(
                    op, builder, mergedOutputs, mergedInputs, mergeInOutDims,
                    mergeScanAxis, vectorLength, op.getReverse()))) {
              applyScan(op, builder, mergedOutputs, mergedInputs,
                        mergedInputType, vectorLength, reverse);
            }
            builder.create<scf::YieldOp>(loc);
          });

      // save back to shared memory
      SmallVector<Value, 4> mergedSharedOutputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getResultTypes()[i]);
        auto tensorType =
            RankedTensorType::get(tType.getShape(), outputElemTypes[i],
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        mergedSharedOutputs.push_back(
            storeToSharedMem(rewriter, tag, tensorType, mergedOutputs[i], true,
                             std::make_pair(op.getOperation(), -1),
                             userAnalysis, replaced2Origin));
      }
      // load from shared memory
      for (unsigned i = 0; i < numOutput; ++i) {
        outputs[i] = loadFromSharedMem(
            rewriter, tag, op.getResultTypes()[i], mergedSharedOutputs[i],
            false, lastUsers[i], std::make_pair(nullptr, -1), userAnalysis,
            replaced2Origin);
      }
    } else {
      // only warp scan
      if (!succeeded(applyVectorizationScan(
              op, rewriter, outputs,
              SmallVector<Value, 4>(adaptor.getSrcs().begin(),
                                    adaptor.getSrcs().end()),
              scanInOutDims, scanAxis, vectorLength, op.getReverse()))) {
        applyScan(op, rewriter, outputs,
                  SmallVector<Value, 4>(adaptor.getSrcs().begin(),
                                        adaptor.getSrcs().end()),
                  inputType, vectorLength, op.getReverse());
      }
    }

    SmallVector<Value, 4> finalOutputs;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto output = outputs[i];
      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(op.getResultTypes()[i]));
      if (resultType.getNumElements() !=
          dyn_cast<MemRefType>(output.getType()).getNumElements()) {
        return op.emitOpError("element number mismatch");
      }
      auto [strides, offset] = resultType.getStridesAndOffset();
      output = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultType, output, offset, resultType.getShape(), strides);
      finalOutputs.push_back(output);
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, finalOutputs);
    return success();
  }

private:
  const GCUArch arch = GCUArch::GCU300;
};
} // namespace

void mlir::triton::populateScanOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateDTETagPool &pTagPool) {
  patterns.add<TTScanOpLowering>(converter, patterns.getContext(), userAnalysis,
                                 replaced2Origin, pTagPool);
}
