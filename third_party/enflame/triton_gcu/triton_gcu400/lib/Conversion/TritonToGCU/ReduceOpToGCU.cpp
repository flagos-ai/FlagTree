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

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace {

std::optional<vector::CombiningKind>
matchReduceCombiningKind(Region &combineOp) {
  auto &opList = combineOp.front().getOperations();
  if (opList.size() != 2) {
    return std::nullopt;
  }
  auto elementWiseOp = &opList.front();
  return TypeSwitch<Operation *, std::optional<vector::CombiningKind>>(
             elementWiseOp)
      .Case<triton::ExternElementwiseOp>(
          [&](auto externElementwiseOp)
              -> std::optional<vector::CombiningKind> {
            auto symbol = externElementwiseOp.getSymbol();
            if (symbol == "__nv_fmaxf") {
              return vector::CombiningKind::MAXIMUMF;
            } else if (symbol == "__nv_max") {
              return vector::CombiningKind::MAXSI;
            } else if (symbol == "__nv_umax") {
              return vector::CombiningKind::MAXUI;
            } else if (symbol == "__nv_fminf") {
              return vector::CombiningKind::MINIMUMF;
            } else if (symbol == "__nv_min") {
              return vector::CombiningKind::MINSI;
            } else if (symbol == "__nv_umin") {
              return vector::CombiningKind::MINUI;
            } else {
              return std::nullopt;
            }
          })
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return vector::CombiningKind::ADD; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return vector::CombiningKind::MUL; })
      .Case<arith::MaxSIOp>(
          [&](auto op) { return vector::CombiningKind::MAXSI; })
      .Case<arith::MaxUIOp>(
          [&](auto op) { return vector::CombiningKind::MAXUI; })
      .Case<arith::MaxNumFOp>(
          [&](auto op) { return vector::CombiningKind::MAXNUMF; })
      .Case<arith::MaximumFOp>(
          [&](auto op) { return vector::CombiningKind::MAXIMUMF; })
      .Case<arith::MinSIOp>(
          [&](auto op) { return vector::CombiningKind::MINSI; })
      .Case<arith::MinUIOp>(
          [&](auto op) { return vector::CombiningKind::MINUI; })
      .Case<arith::MinNumFOp>(
          [&](auto op) { return vector::CombiningKind::MINNUMF; })
      .Case<arith::MinimumFOp>(
          [&](auto op) { return vector::CombiningKind::MINIMUMF; })
      .Case<arith::AndIOp>([&](auto op) { return vector::CombiningKind::AND; })
      .Case<arith::OrIOp>([&](auto op) { return vector::CombiningKind::OR; })
      .Case<arith::XOrIOp>([&](auto op) { return vector::CombiningKind::XOR; })
      .Default([&](auto op) { return std::nullopt; });
}

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
      llvm::cast<triton::ReduceReturnOp>(combineOp.back().getTerminator())
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
  builder.create<triton::ReduceReturnOp>(loc, operands);
}

struct TTReduceOpLowering : SharedConversionPattern<triton::ReduceOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto axis = op.getAxis();
    SmallVector<Value, 4> outputs;
    SmallVector<bool, 4> isSingleElements;
    SmallVector<Type, 4> elemTypes;
    auto numOutput = op.getResults().size();
    auto inputType = dyn_cast<TensorType>(op.getSrcs()[0].getType());
    auto numElems = triton::gcu::getElemsPerThread(inputType);
    SmallVector<int64_t> outputShape(numElems.begin(), numElems.end());
    outputShape[axis] = 1;

    for (unsigned i = 0; i < numOutput; ++i) {
      auto resultType = getTypeConverter()->convertType(op.getType(i));
      bool isSingleElement = !isa<MemRefType>(resultType);
      isSingleElements.push_back(isSingleElement);
      auto elemType = isSingleElement
                          ? resultType
                          : dyn_cast<MemRefType>(resultType).getElementType();
      elemTypes.push_back(elemType);
      auto resultMemRefType = MemRefType::get(outputShape, elemType);

      auto lastUser = userAnalysis.getLastUser(op.getResults()[i]);
      Value output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                 replaced2Origin, resultMemRefType);
      outputs.push_back(output);
    }

    std::array<int64_t, 3> reduceInputDims = {1, 1, 1};
    std::array<int64_t, 3> reduceOutputDims = {1, 1, 1};
    int64_t reduceAxis = 2;
    for (int i = numElems.size() - 1, j = 2; i >= 0; i--) {
      if (static_cast<unsigned>(i) == axis) {
        if (reduceInputDims[j] == 1) {
          reduceInputDims[j] = numElems[i];
        } else {
          reduceInputDims[--j] = numElems[i];
        }
        reduceAxis = j;
        reduceOutputDims[reduceAxis] = 1;
        --j;
      } else {
        reduceInputDims[j] *= numElems[i];
        reduceOutputDims[j] = reduceInputDims[j];
      }
    }
    assert(reduceAxis == 1 || reduceAxis == 2);

    SmallVector<Value, 4> reduceInputs;
    SmallVector<Value, 4> reduceOutputs;
    llvm::transform(
        adaptor.getSrcs(), std::back_inserter(reduceInputs), [&](auto input) {
          return rewriter.create<memref::ReinterpretCastOp>(
              loc,
              MemRefType::get(
                  reduceInputDims,
                  cast<MemRefType>(input.getType()).getElementType()),
              input, ValueRange{}, ValueRange{}, ValueRange{},
              ArrayRef<int64_t>{0}, ArrayRef<int64_t>{reduceInputDims},
              ArrayRef<int64_t>{reduceInputDims[1] * reduceInputDims[2],
                                reduceInputDims[2], 1});
        });
    llvm::transform(
        outputs, std::back_inserter(reduceOutputs), [&](auto output) {
          return rewriter.create<memref::ReinterpretCastOp>(
              loc,
              MemRefType::get(
                  reduceOutputDims,
                  cast<MemRefType>(output.getType()).getElementType()),
              output, ValueRange{}, ValueRange{}, ValueRange{},
              ArrayRef<int64_t>{0}, ArrayRef<int64_t>{reduceOutputDims},
              ArrayRef<int64_t>{reduceOutputDims[1] * reduceOutputDims[2],
                                reduceOutputDims[2], 1});
        });

    applyReduce(op, rewriter, reduceOutputs, reduceInputs, reduceInputDims,
                reduceAxis);
    auto slicedAxies = getSlicedAxies(inputType);
    if (slicedAxies.count(axis) != 0) {
      SmallVector<int64_t> sharedMemShape(inputType.getShape());
      auto encodingAttr = dyn_cast<RankedTensorType>(inputType).getEncoding();
      // use gcu triton::gcu::getWarpsPerCTA
      auto warpsPerCTA = triton::gcu::getWarpsPerCTA(encodingAttr);
      if (warpsPerCTA.size() != sharedMemShape.size()) {
        op.dump();
        assert(false && "the reduce input layout is not a blockencoding!");
      }

      if (warpsPerCTA[axis] < sharedMemShape[axis]) {
        sharedMemShape[axis] = warpsPerCTA[axis];
      }

      bool isReduce1D =
          sharedMemShape[axis] == std::accumulate(sharedMemShape.begin(),
                                                  sharedMemShape.end(), 1,
                                                  std::multiplies<unsigned>());
      triton::gcu::TagInfo tag;
      if (!isReduce1D) {
        tag = pTagPool.getSyncTagInfo(op);
      }
      SmallVector<Value, 4> sharedBuffers;
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      for (unsigned i = 0; i < numOutput; ++i) {
        auto sharedMemRefType =
            MemRefType::get(sharedMemShape, elemTypes[i], AffineMap{},
                            rewriter.getI64IntegerAttr(2) /*shared memory*/);
        sharedBuffers.emplace_back(
            rewriter.create<memref::AllocOp>(loc, sharedMemRefType));
        if (isReduce1D) {
          rewriter.create<memref::StoreOp>(
              loc,
              rewriter.create<memref::LoadOp>(loc, reduceOutputs[i],
                                              ValueRange{zero, zero, zero}),
              sharedBuffers.back(),
              ValueRange{getWarpIds(rewriter, loc, inputType)});
          rewriter.create<gpu::BarrierOp>(loc);
        } else {
          storeToSharedMem(rewriter, tag, inputType, sharedBuffers.back(),
                           outputs[i], false);
        }
      }

      if (warpsPerCTA[axis] < sharedMemShape[axis]) {
        reduceInputDims[reduceAxis] = warpsPerCTA[axis];
      } else {
        reduceInputDims[reduceAxis] = sharedMemShape[axis];
      }
      auto loadFromShareForAllReduce =
          [&](OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
              Value buffer, triton::gcu::FirstLastUserAnalysis &userAnalysis,
              std::map<Operation *, Operation *> &replaced2Origin) {
            auto loc = buffer.getLoc();
            auto srcType = dyn_cast<MemRefType>(buffer.getType());
            auto numElems = triton::gcu::getElemsPerThread(type);
            numElems[axis] = warpsPerCTA[axis];
            auto totalNumElems = builder.create<arith::ConstantIndexOp>(
                loc, std::accumulate(numElems.begin(), numElems.end(), 1,
                                     std::multiplies<unsigned>()));
            auto outputType = MemRefType::get(
                SmallVector<int64_t>(numElems.begin(), numElems.end()),
                srcType.getElementType());
            auto warpIds = getWarpIds(builder, loc, type);
            SmallVector<Value, 4> offsets;
            for (unsigned i = 0; i < srcType.getRank(); ++i) {
              if (i == axis) {
                offsets.push_back(
                    builder.create<arith::ConstantIntOp>(loc, 0, 32));
              } else {
                offsets.push_back(builder.create<arith::MulIOp>(
                    loc,
                    builder.create<arith::ConstantIntOp>(loc, numElems[i], 32),
                    builder.create<arith::IndexCastOp>(
                        loc, builder.getI32Type(), warpIds[i])));
              }
            }
            auto output =
                syncAllocOp(builder, loc, std::make_pair(op.getOperation(), -1),
                            userAnalysis, replaced2Origin, outputType);
            auto defaultValue = triton::gcu::createConstantZero(
                builder, loc, srcType.getElementType());
            if (srcType.getRank() > 5) {
              SmallVector<Value, 4> mergedOffsets;
              Value src;
              Value dst;
              mergeContinuousDims(builder, loc, src, dst, offsets,
                                  mergedOffsets, srcType, outputType, buffer,
                                  output);
              builder.create<memref_ext::SliceStartOp>(
                  loc, dst, src, mergedOffsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
              auto [oriOutputStrides, oriOutputOffset] =
                  outputType.getStridesAndOffset();
              builder.create<memref::ReinterpretCastOp>(
                  loc, outputType, dst, oriOutputOffset,
                  SmallVector<int64_t>(numElems.begin(), numElems.end()),
                  oriOutputStrides);
            } else {
              builder.create<memref_ext::SliceStartOp>(
                  loc, output, buffer, offsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
            }
            builder.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
            return output;
          };

      SmallVector<Value, 4> warpReduceInputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        if (isReduce1D) {
          warpReduceInputs.push_back(sharedBuffers[i]);
        } else {
          auto tensorType =
              RankedTensorType::get(sharedMemShape, elemTypes[i], encodingAttr);
          warpReduceInputs.emplace_back(loadFromShareForAllReduce(
              rewriter, tag, tensorType, sharedBuffers[i], userAnalysis,
              replaced2Origin));
        }
      }

      llvm::transform(
          warpReduceInputs, warpReduceInputs.begin(), [&](auto input) {
            return rewriter.create<memref::ReinterpretCastOp>(
                loc,
                MemRefType::get(
                    reduceInputDims,
                    cast<MemRefType>(input.getType()).getElementType(),
                    MemRefLayoutAttrInterface{},
                    isReduce1D ? rewriter.getI64IntegerAttr(2) : Attribute{}),
                input, ValueRange{}, ValueRange{}, ValueRange{},
                ArrayRef<int64_t>{0}, ArrayRef<int64_t>{reduceInputDims},
                ArrayRef<int64_t>{reduceInputDims[1] * reduceInputDims[2],
                                  reduceInputDims[2], 1});
          });
      applyReduce(op, rewriter, reduceOutputs, warpReduceInputs,
                  reduceInputDims, reduceAxis);
      for (auto buffer : sharedBuffers) {
        rewriter.create<memref::DeallocOp>(loc, buffer);
      }
    }

    SmallVector<Value, 4> finalOutputs;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto output = outputs[i];
      if (isSingleElements[i]) {
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        output = rewriter.create<memref::LoadOp>(loc, output, ValueRange{zero});
      } else {
        auto resultType = dyn_cast<MemRefType>(
            getTypeConverter()->convertType(op.getResultTypes()[i]));
        if (resultType.getNumElements() !=
            dyn_cast<MemRefType>(output.getType()).getNumElements()) {
          return op.emitOpError("element number mismatch: ")
                 << resultType.getNumElements() << " vs "
                 << dyn_cast<MemRefType>(output.getType()).getNumElements();
        }
        auto [strides, offset] = resultType.getStridesAndOffset();
        output = rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, output, offset, resultType.getShape(), strides);
      }
      finalOutputs.push_back(output);
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, finalOutputs);
    return success();
  }

private:
  void applyReduce(triton::ReduceOp op, OpBuilder &rewriter,
                   ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                   const std::array<int64_t, 3> &reduceDims,
                   int64_t reduceAxis) const {
    if (succeeded(applyVectorizationImpl(op, rewriter, outputs, inputs,
                                         reduceDims, reduceAxis))) {
      return;
    }
    applyScalarImpl(op, rewriter, outputs, inputs, reduceDims, reduceAxis);
  }

  void applyScalarImpl(triton::ReduceOp op, OpBuilder &rewriter,
                       ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                       const std::array<int64_t, 3> &reduceDims,
                       int64_t reduceAxis) const {
    auto loc = op.getLoc();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto numOutput = outputs.size();
    rewriter.create<scf::ForOp>(
        loc, zero, rewriter.create<arith::ConstantIndexOp>(loc, reduceDims[0]),
        one, ValueRange{},
        [&](OpBuilder &builder, Location loc, Value iter0,
            ValueRange iterArgs) {
          builder.create<scf::ForOp>(
              loc, zero,
              builder.create<arith::ConstantIndexOp>(
                  loc, reduceDims[3 - reduceAxis]),
              one, ValueRange{},
              [&](OpBuilder &builder, Location loc, Value iter1,
                  ValueRange iterArgs) {
                SmallVector<Value> iterators(3);
                iterators[0] = iter0;
                iterators[3 - reduceAxis] = iter1;
                iterators[reduceAxis] = zero;
                SmallVector<Value> initValues;
                llvm::transform(inputs, std::back_inserter(initValues),
                                [&](auto input) {
                                  return builder.create<memref::LoadOp>(
                                      loc, input, iterators);
                                });
                auto loop = builder.create<scf::ForOp>(
                    loc, one,
                    builder.create<arith::ConstantIndexOp>(
                        loc, reduceDims[reduceAxis]),
                    one, initValues,
                    [&](OpBuilder &builder, Location loc, Value iter2,
                        ValueRange iterArgs) {
                      SmallVector<Value, 4> operands(iterArgs.begin(),
                                                     iterArgs.end());
                      SmallVector<Type> resultElemTypes;
                      iterators[reduceAxis] = iter2;
                      for (unsigned i = 0; i < numOutput; ++i) {
                        operands.push_back(builder.create<memref::LoadOp>(
                            loc, inputs[i], iterators));
                        resultElemTypes.push_back(operands.back().getType());
                      }
                      auto executeRegionOp =
                          builder.create<scf::ExecuteRegionOp>(loc,
                                                               resultElemTypes);
                      executeRegionOp.getRegion().emplaceBlock();
                      IRMapping map;
                      for (auto [arg, operand] : llvm::zip(
                               op.getCombineOp().getArguments(), operands)) {
                        map.map(arg, operand);
                      }
                      {
                        OpBuilder::InsertionGuard guard(builder);
                        builder.setInsertionPointToStart(
                            &executeRegionOp.getRegion().getBlocks().back());
                        for (auto &o : op.getCombineOp().getBlocks().back()) {
                          auto newOp = builder.clone(o, map);
                          for (auto [result, newResult] :
                               llvm::zip(o.getResults(), newOp->getResults())) {
                            map.map(result, newResult);
                          }
                        }
                      }
                      builder.create<scf::YieldOp>(
                          loc, executeRegionOp.getResults());
                    });
                iterators[reduceAxis] = zero;
                for (unsigned i = 0; i < numOutput; ++i) {
                  builder.create<memref::StoreOp>(loc, loop.getResult(i),
                                                  outputs[i], iterators);
                }
                builder.create<scf::YieldOp>(loc);
              });
          builder.create<scf::YieldOp>(loc);
        });
    doMemFence(rewriter, op);
  }

  LogicalResult applyVectorizationImpl(triton::ReduceOp op, OpBuilder &rewriter,
                                       ArrayRef<Value> outputs,
                                       ArrayRef<Value> inputs,
                                       const std::array<int64_t, 3> &reduceDims,
                                       int64_t reduceAxis) const {
    SmallVector<Type> elementTypes;
    unsigned maxBpe = 1;
    unsigned minBpe = 8;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32()) {
        // vectorized reduce instruction are not supported for the i64 type
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
      elementTypes.push_back(elementType);
    }
    auto numOacc = maxBpe / minBpe;
    if (numOacc > 4) {
      return failure();
    }
    int64_t vectorizeAxis = 3;
    unsigned vectorLength = oaccSizeInBytes / minBpe;
    for (auto i = 2; i >= 0; --i) {
      if (reduceDims[i] >= vectorLength) {
        vectorizeAxis = i;
        break;
      }
    }

    SmallVector<bool> needPad;
    if (vectorizeAxis == 3) {
      if (reduceAxis == 2) {
        return failure();
      } else {
        for (auto elementType : elementTypes) {
          if (mlir::triton::gcu::getBpe(elementType) * reduceDims[1] >=
              oaccSizeInBytes) {
            needPad.push_back(false);
          } else {
            needPad.push_back(true);
          }
        }
        vectorizeAxis = 2;
      }
    }

    auto loc = op.getLoc();
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<VectorType, 4> vectorTypes;
    constexpr int loopUnrollTime = 16;

    SmallVector<Value, 2> tmpBuffers;
    SmallVector<Value, 2> reduceOutputs(outputs.begin(), outputs.end());
    std::array<int64_t, 3> reduceInputDims = reduceDims;
    std::array<int64_t, 3> reduceOutputDims = reduceInputDims;
    reduceOutputDims[reduceAxis] = 1;

    bool needTranspose = false;
    std::array<int64_t, 3> transposeLayout = {0, 1, 2};
    SmallVector<Value, 3> transposeLayoutValue;
    if (vectorizeAxis != 2) {
      needTranspose = true;
      transposeLayout[vectorizeAxis] = 2;
      transposeLayout[2] = vectorizeAxis;
    }

    if (needTranspose) {
      reduceInputDims[0] = reduceDims[transposeLayout[0]];
      reduceInputDims[1] = reduceDims[transposeLayout[1]];
      reduceInputDims[2] = reduceDims[transposeLayout[2]];
      reduceAxis = transposeLayout[reduceAxis];
      llvm::transform(transposeLayout, std::back_inserter(transposeLayoutValue),
                      [&](auto dim) {
                        return rewriter.create<arith::ConstantIntOp>(loc, dim,
                                                                     32);
                      });
      auto tag = pTagPool.getSyncTagInfo(op);
      llvm::transform(inputs, std::back_inserter(tmpBuffers), [&](auto input) {
        auto memrefTy = cast<MemRefType>(input.getType());
        auto elementTy = memrefTy.getElementType();
        auto tmpBuffer = rewriter.create<memref::AllocOp>(
            loc,
            MemRefType::get(ArrayRef<int64_t>{reduceInputDims}, elementTy));
        rewriter.create<memref_ext::TransposeStartOp>(
            loc, tmpBuffer, input, transposeLayoutValue, tag.getTag(),
            ValueRange{tag.getIdx()});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag.getTag(), ValueRange{tag.getIdx()},
            rewriter.create<arith::ConstantIndexOp>(loc,
                                                    memrefTy.getNumElements()));
        return tmpBuffer;
      });
      inputs = tmpBuffers;
      reduceOutputDims = reduceInputDims;
      reduceOutputDims[reduceAxis] = 1;
      if (vectorizeAxis == 0) {
        assert(reduceAxis == 1);
        llvm::transform(
            elementTypes, reduceOutputs.begin(), [&](auto elementTy) {
              return rewriter.create<memref::AllocOp>(
                  loc, MemRefType::get(ArrayRef<int64_t>{reduceOutputDims},
                                       elementTy));
            });
      } else {
        llvm::transform(outputs, reduceOutputs.begin(), [&](auto output) {
          auto memrefType = MemRefType::get(
              ArrayRef<int64_t>{reduceOutputDims[0], reduceOutputDims[1],
                                reduceOutputDims[2]},
              cast<MemRefType>(output.getType()).getElementType());
          return rewriter.create<memref::ReinterpretCastOp>(
              loc, memrefType, output, 0, ArrayRef<int64_t>{reduceOutputDims},
              ArrayRef<int64_t>{reduceOutputDims[2] * reduceOutputDims[1],
                                reduceOutputDims[2], 1});
        });
        needTranspose = false;
      }
      vectorizeAxis = 2;
    }

    assert(vectorizeAxis == 2);
    assert(reduceAxis == 1 || reduceAxis == 2);
    bool isReduce1D = reduceDims[0] == 1 && reduceDims[1] == 1;
    auto &combineOp = op.getCombineOp();
    for (unsigned i = 0; i < numOutput; ++i) {
      if (cast<MemRefType>(inputs[i].getType()).getElementType().isInteger(1)) {
        elementTypes[i] = rewriter.getIntegerType(8);
      }
    }

    llvm::transform(elementTypes, std::back_inserter(vectorTypes),
                    [vectorLength](auto elementTy) {
                      return VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             elementTy);
                    });
    SmallVector<Value, loopUnrollTime> cur;
    SmallVector<Value, loopUnrollTime> next;
    auto loopLimit = reduceAxis == 1 || !isReduce1D
                         ? reduceInputDims[1]
                         : reduceInputDims[2] / vectorLength;
    auto loopCnt = loopLimit > loopUnrollTime ? loopUnrollTime : loopLimit;
    auto loopCntValue = rewriter.create<arith::ConstantIndexOp>(loc, loopCnt);
    if (!needPad.empty()) {
      SmallVector<Value, 2> reduceInputs(inputs.begin(), inputs.end());
      for (unsigned i = 0; i < numOutput; ++i) {
        auto elementTy = cast<MemRefType>(inputs[i].getType()).getElementType();
        if (elementTy.isInteger(1)) {
          reduceInputs[i] = rewriter.create<memref::ReinterpretCastOp>(
              loc, MemRefType::get(reduceInputDims, rewriter.getIntegerType(8)),
              rewriter.create<mlir::gcu::PtrToMemRefOp>(
                  loc,
                  MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                                  rewriter.getIntegerType(8)),
                  rewriter.create<mlir::gcu::MemRefToPtrOp>(
                      loc,
                      mlir::gcu::PtrType::get(rewriter.getContext(), elementTy),
                      reduceInputs[i])),
              0, ArrayRef<int64_t>{reduceInputDims},
              ArrayRef<int64_t>{reduceInputDims[2] * reduceInputDims[1],
                                reduceInputDims[2], 1});
          reduceOutputs[i] = rewriter.create<memref::ReinterpretCastOp>(
              loc,
              MemRefType::get(reduceOutputDims, rewriter.getIntegerType(8)),
              rewriter.create<mlir::gcu::PtrToMemRefOp>(
                  loc,
                  MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                                  rewriter.getIntegerType(8)),
                  rewriter.create<mlir::gcu::MemRefToPtrOp>(
                      loc,
                      mlir::gcu::PtrType::get(rewriter.getContext(), elementTy),
                      reduceOutputs[i])),
              0, ArrayRef<int64_t>{reduceOutputDims},
              ArrayRef<int64_t>{reduceOutputDims[2] * reduceOutputDims[1],
                                reduceOutputDims[2], 1});
          elementTypes[i] = rewriter.getIntegerType(8);
        }
      }
      inputs = reduceInputs;
      auto mask = rewriter.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          rewriter.getI1Type()),
          DenseI64ArrayAttr::get(rewriter.getContext(),
                                 ArrayRef<int64_t>{reduceOutputDims[2]}));
      rewriter.create<scf::ForOp>(
          loc, zero,
          rewriter.create<arith::ConstantIndexOp>(loc, reduceInputDims[0]), one,
          ValueRange{},
          [&](OpBuilder &builder, Location loc, Value iter0,
              ValueRange iterArgs) {
            for (unsigned i = 0; i < loopCnt; ++i) {
              for (unsigned j = 0; j < numOutput; ++j) {
                if (needPad[j]) {
                  cur.emplace_back(builder.create<vector::MaskedLoadOp>(
                      loc, vectorTypes[j], inputs[j],
                      ValueRange{iter0,
                                 builder.create<arith::ConstantIndexOp>(loc, i),
                                 zero},
                      mask,
                      builder.create<arith::ConstantOp>(
                          loc, DenseElementsAttr::get(
                                   vectorTypes[j],
                                   builder.getZeroAttr(elementTypes[j])))));
                } else {
                  cur.emplace_back(builder.create<vector::LoadOp>(
                      loc, vectorTypes[j], inputs[j],
                      ValueRange{iter0,
                                 builder.create<arith::ConstantIndexOp>(loc, i),
                                 zero}));
                }
              }
            }
            auto loop = builder.create<scf::ForOp>(
                loc, loopCntValue,
                builder.create<arith::ConstantIndexOp>(loc, reduceInputDims[1]),
                loopCntValue, cur,
                [&](OpBuilder &builder, Location loc, Value iter,
                    ValueRange iterArgs) {
                  next.resize(loopCnt * numOutput);
                  for (unsigned i = 0; i < loopCnt; ++i) {
                    for (unsigned j = 0; j < numOutput; ++j) {
                      if (needPad[j]) {
                        next[i * numOutput + j] =
                            builder.create<vector::MaskedLoadOp>(
                                loc, vectorTypes[j], inputs[j],
                                ValueRange{
                                    iter0,
                                    builder.create<arith::AddIOp>(
                                        loc,
                                        builder.create<arith::ConstantIndexOp>(
                                            loc, i),
                                        iter),
                                    zero},
                                mask,
                                builder.create<arith::ConstantOp>(
                                    loc,
                                    DenseElementsAttr::get(
                                        vectorTypes[j],
                                        builder.getZeroAttr(elementTypes[j]))));
                      } else {
                        next[i * numOutput + j] =
                            builder.create<vector::LoadOp>(
                                loc, vectorTypes[j], inputs[j],
                                ValueRange{
                                    iter0,
                                    builder.create<arith::AddIOp>(
                                        loc,
                                        builder.create<arith::ConstantIndexOp>(
                                            loc, i),
                                        iter),
                                    zero});
                      }
                    }
                  }
                  SmallVector<Value, 4> args(numOutput * 2);
                  SmallVector<Value, loopUnrollTime> terminatorOperands;
                  for (unsigned i = 0; i < loopCnt; ++i) {
                    for (unsigned j = 0; j < numOutput; ++j) {
                      args[j] = iterArgs[i * numOutput + j];
                      args[numOutput + j] = next[i * numOutput + j];
                    }
                    terminatorOperands.append(
                        vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp, args, vectorLength));
                  }
                  vectorizeCombineOpTerminator(loc, builder,
                                               terminatorOperands);
                });
            cur.reserve(cur.size() * 2);
            cur.assign(loop.getResults().begin(), loop.getResults().end());
            auto iter = cur.begin();
            while (loopCnt != 1) {
              loopCnt /= 2;
              for (auto i = 0; i < loopCnt; ++i) {
                cur.append(vectorizeCombineOpWithoutTerminator(
                    loc, builder, combineOp, ValueRange(iter, 2 * numOutput),
                    vectorLength));
                iter = std::next(iter, 2 * numOutput);
              }
            }
            for (unsigned i = 0; i < numOutput; ++i) {
              if (needPad[i]) {
                builder.create<vector::MaskedStoreOp>(
                    loc, reduceOutputs[i], ValueRange{iter0, zero, zero}, mask,
                    *iter++);
              } else {
                builder.create<vector::StoreOp>(loc, *iter++, reduceOutputs[i],
                                                ValueRange{iter0, zero, zero});
              }
            }
            builder.create<scf::YieldOp>(loc);
          });
      return success();
    }

    SmallVector<Value> tarAddrs;
    SmallVector<SmallVector<Value>> tarStrides(numOutput);
    triton::gcu::TritonGCUBuilder b(loc, rewriter);
    for (unsigned i = 0; i < numOutput; ++i) {
      // input address
      tarAddrs.emplace_back(b.tarAddr(inputs[i]));
    }
    if (reduceAxis == 1) {
      for (unsigned i = 0; i < numOutput; ++i) {
        // output address
        tarAddrs.emplace_back(b.tarAddr(reduceOutputs[i]));
        auto bpe = mlir::triton::gcu::getBpe(elementTypes[i]);
        // input stride for looping over dim1
        tarStrides[i].emplace_back(
            b.tarStride(vectorTypes[i], reduceInputDims[2] * bpe));
        // input stride for looping over dim2
        tarStrides[i].emplace_back(b.tarValue(
            (vectorLength - reduceInputDims[1] * reduceInputDims[2]) * bpe));
        // input stride for looping over dim0
        tarStrides[i].emplace_back(
            b.tarValue((reduceInputDims[1] - 1) * reduceInputDims[2] * bpe));
        // input stride for looping over dim2
        tarStrides[i].emplace_back(b.tarValue(vectorLength * bpe));
      }
      rewriter.create<scf::ForOp>(
          loc, zero,
          rewriter.create<arith::ConstantIndexOp>(loc, reduceInputDims[0]), one,
          tarAddrs,
          [&](OpBuilder &builder, Location loc, Value iter2,
              ValueRange iterArgs2) {
            SmallVector<Value> initArgs(iterArgs2);
            auto loop1 = builder.create<scf::ForOp>(
                loc, zero,
                builder.create<arith::ConstantIndexOp>(loc, reduceInputDims[2]),
                builder.create<arith::ConstantIndexOp>(loc, vectorLength),
                initArgs,
                [&](OpBuilder &builder, Location loc, Value iter1,
                    ValueRange iterArgs1) {
                  SmallVector<Value> initArgs(iterArgs1.take_front(numOutput));
                  for (unsigned i = 0; i < loopCnt; ++i) {
                    for (unsigned j = 0; j < numOutput; ++j) {
                      cur.emplace_back(b.tarLoad(vectorTypes[j], initArgs[j],
                                                 tarStrides[j][0]));
                    }
                  }
                  initArgs.append(cur);
                  auto loop0 = builder.create<scf::ForOp>(
                      loc, loopCntValue,
                      builder.create<arith::ConstantIndexOp>(
                          loc, reduceInputDims[1]),
                      loopCntValue, initArgs,
                      [&](OpBuilder &builder, Location loc, Value iter0,
                          ValueRange iterArgs0) {
                        SmallVector<Value> inputAddrs(
                            iterArgs0.take_front(numOutput));
                        for (unsigned i = 0; i < loopCnt; ++i) {
                          for (unsigned j = 0; j < numOutput; ++j) {
                            next.emplace_back(b.tarLoad(vectorTypes[j],
                                                        inputAddrs[j],
                                                        tarStrides[j][0]));
                          }
                        }
                        SmallVector<Value, 4> args(numOutput * 2);
                        SmallVector<Value> terminatorOperands(inputAddrs);
                        for (unsigned i = 0; i < loopCnt; ++i) {
                          for (unsigned j = 0; j < numOutput; ++j) {
                            args[j] = iterArgs0[numOutput + i * numOutput + j];
                            args[numOutput + j] = next[i * numOutput + j];
                          }
                          terminatorOperands.append(
                              vectorizeCombineOpWithoutTerminator(
                                  loc, builder, combineOp, args, vectorLength));
                        }
                        vectorizeCombineOpTerminator(loc, builder,
                                                     terminatorOperands);
                      });
                  cur.reserve(cur.size() * 2);
                  cur.assign(loop0.getResults().begin() + numOutput,
                             loop0.getResults().end());
                  auto iter = cur.begin();
                  while (loopCnt != 1) {
                    loopCnt /= 2;
                    for (auto i = 0; i < loopCnt; ++i) {
                      cur.append(vectorizeCombineOpWithoutTerminator(
                          loc, builder, combineOp,
                          ValueRange(iter, 2 * numOutput), vectorLength,
                          loopCnt == 1));
                      iter = std::next(iter, 2 * numOutput);
                    }
                  }
                  SmallVector<Value> outputAddrs(
                      iterArgs1.slice(numOutput, numOutput));
                  SmallVector<Value> terminatorOperands(
                      loop0.getResults().begin(),
                      loop0.getResults().begin() + numOutput);

                  for (unsigned i = 0; i < numOutput; ++i) {
                    b.tarJump(terminatorOperands[i], tarStrides[i][1]);
                    b.tarStore(*iter++, outputAddrs[i], tarStrides[i][3]);
                  }
                  terminatorOperands.append(outputAddrs);
                  builder.create<scf::YieldOp>(loc, terminatorOperands);
                });
            SmallVector<Value> terminatorOperands(loop1.getResults());
            for (unsigned i = 0; i < numOutput; ++i) {
              b.tarJump(terminatorOperands[i], tarStrides[i][2]);
            }
            builder.create<scf::YieldOp>(loc, terminatorOperands);
          });
    } else {
      // reduceAxis == 2
      if (isReduce1D) {
        for (unsigned i = 0; i < numOutput; ++i) {
          auto bpe = mlir::triton::gcu::getBpe(elementTypes[i]);
          // input stride
          tarStrides[i].emplace_back(
              b.tarStride(vectorTypes[i], vectorLength * bpe));
        }
        for (unsigned i = 0; i < loopCnt; ++i) {
          for (unsigned j = 0; j < numOutput; ++j) {
            cur.emplace_back(
                b.tarLoad(vectorTypes[j], tarAddrs[j], tarStrides[j][0]));
          }
        }
        SmallVector<Value> initArgs(tarAddrs);
        initArgs.append(cur);
        auto loop = rewriter.create<scf::ForOp>(
            loc,
            rewriter.create<arith::ConstantIndexOp>(loc,
                                                    loopCnt * vectorLength),
            rewriter.create<arith::ConstantIndexOp>(loc, reduceInputDims[2]),
            rewriter.create<arith::ConstantIndexOp>(loc,
                                                    loopCnt * vectorLength),
            initArgs,
            [&](OpBuilder &builder, Location loc, Value iter,
                ValueRange iterArgs) {
              SmallVector<Value> address(iterArgs.take_front(numOutput));
              for (unsigned i = 0; i < loopCnt; ++i) {
                for (unsigned j = 0; j < numOutput; ++j) {
                  next.emplace_back(
                      b.tarLoad(vectorTypes[j], address[j], tarStrides[j][0]));
                }
              }
              SmallVector<Value, 4> args(numOutput * 2);
              SmallVector<Value> terminatorOperands(address);
              for (unsigned i = 0; i < loopCnt; ++i) {
                for (unsigned j = 0; j < numOutput; ++j) {
                  args[j] = iterArgs[numOutput + i * numOutput + j];
                  args[numOutput + j] = next[i * numOutput + j];
                }
                terminatorOperands.append(vectorizeCombineOpWithoutTerminator(
                    loc, builder, combineOp, args, vectorLength));
              }
              vectorizeCombineOpTerminator(loc, builder, terminatorOperands);
            });
        cur.reserve(cur.size() * 2);
        cur.assign(loop.getResults().begin() + numOutput,
                   loop.getResults().end());
        auto iter = cur.begin();
        while (loopCnt != 1) {
          loopCnt /= 2;
          for (unsigned i = 0; i < loopCnt; ++i) {
            cur.append(vectorizeCombineOpWithoutTerminator(
                loc, rewriter, combineOp, ValueRange(iter, 2 * numOutput),
                vectorLength));
            iter = std::next(iter, 2 * numOutput);
          }
        }
        auto results = vReduce(loc, rewriter, combineOp,
                               ValueRange(iter, numOutput), vectorLength);
        for (unsigned i = 0; i < numOutput; ++i) {
          if (cast<MemRefType>(reduceOutputs[i].getType())
                  .getElementType()
                  .isInteger(1)) {
            results[i] = rewriter.create<arith::TruncIOp>(
                loc, rewriter.getI1Type(), results[i]);
          }
          rewriter.create<memref::StoreOp>(loc, results[i], reduceOutputs[i],
                                           ValueRange{zero, zero, zero});
        }
      } else {
        rewriter.create<scf::ForOp>(
            loc, zero,
            rewriter.create<arith::ConstantIndexOp>(loc, reduceInputDims[0]),
            one, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter2,
                ValueRange iterArgs2) {
              for (unsigned i = 0; i < numOutput; ++i) {
                auto bpe = mlir::triton::gcu::getBpe(elementTypes[i]);
                // input stride for looping over dim2
                tarStrides[i].emplace_back(
                    b.tarStride(vectorTypes[i], reduceInputDims[2] * bpe));
                // back stride for looping over dim2
                tarStrides[i].emplace_back(b.tarStride(
                    vectorTypes[i],
                    (vectorLength - (loopCnt - 1) * reduceInputDims[2]) * bpe));
                // input stride for looping over dim1
                tarStrides[i].emplace_back(
                    b.tarValue((loopCnt - 1) * reduceInputDims[2] * bpe));
              }
              builder.create<scf::ForOp>(
                  loc, zero,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         reduceInputDims[1]),
                  loopCntValue, tarAddrs,
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs1) {
                    SmallVector<Value> initArgs(iterArgs1);
                    for (unsigned i = 0; i < loopCnt - 1; ++i) {
                      for (unsigned j = 0; j < numOutput; ++j) {
                        cur.emplace_back(b.tarLoad(vectorTypes[j], initArgs[j],
                                                   tarStrides[j][0]));
                      }
                    }
                    for (unsigned i = 0; i < numOutput; ++i) {
                      cur.emplace_back(b.tarLoad(vectorTypes[i], initArgs[i],
                                                 tarStrides[i][1]));
                    }
                    initArgs.append(cur);
                    auto loop0 = builder.create<scf::ForOp>(
                        loc,
                        builder.create<arith::ConstantIndexOp>(loc,
                                                               vectorLength),
                        builder.create<arith::ConstantIndexOp>(
                            loc, reduceInputDims[2]),
                        builder.create<arith::ConstantIndexOp>(loc,
                                                               vectorLength),
                        initArgs,
                        [&](OpBuilder &builder, Location loc, Value iter0,
                            ValueRange iterArgs0) {
                          SmallVector<Value> address(
                              iterArgs0.take_front(tarAddrs.size()));
                          for (unsigned i = 0; i < loopCnt - 1; ++i) {
                            for (unsigned j = 0; j < numOutput; ++j) {
                              next.emplace_back(b.tarLoad(vectorTypes[j],
                                                          address[j],
                                                          tarStrides[j][0]));
                            }
                          }
                          for (unsigned i = 0; i < numOutput; ++i) {
                            next.emplace_back(b.tarLoad(
                                vectorTypes[i], address[i], tarStrides[i][1]));
                          }

                          SmallVector<Value, 4> args(numOutput * 2);
                          SmallVector<Value> terminatorOperands(address);
                          for (unsigned i = 0; i < loopCnt; ++i) {
                            for (unsigned j = 0; j < numOutput; ++j) {
                              args[j] =
                                  iterArgs0[numOutput + i * numOutput + j];
                              args[numOutput + j] = next[i * numOutput + j];
                            }
                            terminatorOperands.append(
                                vectorizeCombineOpWithoutTerminator(
                                    loc, builder, combineOp, args,
                                    vectorLength));
                          }
                          vectorizeCombineOpTerminator(loc, builder,
                                                       terminatorOperands);
                        });
                    for (unsigned i = 0; i < numOutput; ++i) {
                      initArgs[i] = loop0.getResult(i);
                      b.tarJump(initArgs[i], tarStrides[i][2]);
                    }
                    for (unsigned i = 0; i < loopCnt; ++i) {
                      for (unsigned j = 0; j < numOutput; ++j) {
                        auto results =
                            vReduce(loc, builder, combineOp,
                                    ValueRange(loop0.getResults().slice(
                                        numOutput + i * numOutput, numOutput)),
                                    vectorLength);
                        if (cast<MemRefType>(reduceOutputs[j].getType())
                                .getElementType()
                                .isInteger(1)) {
                          results[j] = rewriter.create<arith::TruncIOp>(
                              loc, rewriter.getI1Type(), results[j]);
                        }
                        builder.create<memref::StoreOp>(
                            loc, results[j], reduceOutputs[j],
                            ValueRange{
                                iter2,
                                builder.create<arith::AddIOp>(
                                    loc,
                                    builder.create<arith::ConstantIndexOp>(loc,
                                                                           i),
                                    iter1),
                                zero});
                      }
                    }
                    builder.create<scf::YieldOp>(
                        loc, ValueRange(initArgs.begin(), numOutput));
                  });
              builder.create<scf::YieldOp>(loc);
            });
      }
    }

    for (auto buffer : tmpBuffers) {
      rewriter.create<memref::DeallocOp>(loc, buffer);
    }

    if (needTranspose) {
      for (unsigned i = 0; i < numOutput; ++i) {
        auto memrefTy = cast<MemRefType>(outputs[i].getType());
        auto tag = pTagPool.getSyncTagInfo(op);
        rewriter.create<memref_ext::TransposeStartOp>(
            loc, outputs[i], reduceOutputs[i], transposeLayoutValue,
            tag.getTag(), ValueRange{tag.getIdx()});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag.getTag(), ValueRange{tag.getIdx()},
            rewriter.create<arith::ConstantIndexOp>(loc,
                                                    memrefTy.getNumElements()));
        rewriter.create<memref::DeallocOp>(loc, reduceOutputs[i]);
      }
    }
    return success();
  }

  SmallVector<Value> vReduce(Location loc, OpBuilder &builder,
                             Region &combineOp, ValueRange vecValues,
                             int64_t vectorLength) const {
    assert(llvm::all_of(vecValues.getTypes(), [&](auto ty) {
      auto vecTy = dyn_cast<VectorType>(ty);
      return vecTy && vecTy.getRank() == 1 &&
             vecTy.getDimSize(0) == vectorLength;
    }));

    if (auto kind = matchReduceCombiningKind(combineOp)) {
      assert(vecValues.size() == 1);
      return {builder.create<vector::ReductionOp>(loc, *kind, vecValues[0])};
    }

    SmallVector<Value> calValues;
    SmallVector<Type, 4> resultElemTypes;
    for (auto v : vecValues) {
      calValues.emplace_back(builder.create<vector::ExtractOp>(loc, v, 0));
      resultElemTypes.push_back(calValues.back().getType());
    }
    auto loop = builder.create<scf::ForOp>(
        loc, builder.create<arith::ConstantIndexOp>(loc, 1),
        builder.create<arith::ConstantIndexOp>(loc, vectorLength),
        builder.create<arith::ConstantIndexOp>(loc, 1), calValues,
        [&](OpBuilder &builder, Location loc, Value iter, ValueRange iterArgs) {
          SmallVector<Value, 4> operands(iterArgs.begin(), iterArgs.end());
          for (auto v : vecValues) {
            operands.emplace_back(
                builder.create<vector::ExtractOp>(loc, v, iter));
          }
          auto executeRegionOp =
              builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
          executeRegionOp.getRegion().emplaceBlock();
          IRMapping map;
          for (auto [arg, operand] :
               llvm::zip(combineOp.getArguments(), operands)) {
            map.map(arg, operand);
          }
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(
                &executeRegionOp.getRegion().getBlocks().back());
            for (auto &o : combineOp.getBlocks().back()) {
              auto newO = builder.clone(o, map);
              for (auto [result, newResult] :
                   llvm::zip(o.getResults(), newO->getResults())) {
                map.map(result, newResult);
              }
            }
          }
          builder.create<scf::YieldOp>(
              loc, ValueRange{executeRegionOp.getResults()});
        });
    return loop.getResults();
  }
};
} // namespace

void mlir::triton::populateReduceOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<TTReduceOpLowering>(converter, patterns.getContext(),
                                   userAnalysis, replaced2Origin, pTagPool);
}
