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

template <typename OP>
bool patternMatch(Block::iterator beg, Block::iterator end) {
  if (beg == end)
    return false;
  return isa<OP>(&*beg) && ++beg == end;
}

template <typename OP0, typename OP1, typename... OPN>
bool patternMatch(Block::iterator beg, Block::iterator end) {
  if (beg == end)
    return false;
  auto ret = isa<OP0>(&*beg);
  return ret ? patternMatch<OP1, OPN...>(++beg, end) : ret;
}

template <typename OP0, typename OP1, typename... OPN>
bool patternMatch(Block::OpListType *opList) {
  return patternMatch<OP0, OP1, OPN...>(opList->begin(), opList->end());
}

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
    auto loc = op.getLoc();
    auto axis = op.getAxis();
    SmallVector<Value, 4> outputs;
    SmallVector<bool, 4> isSingleElements;
    SmallVector<Type, 4> elemTypes;
    auto numOutput = op.getResults().size();
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
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
      Value tag;
      if (!isReduce1D) {
        tag = getPrivateDTETag(rewriter, op);
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
          [&](OpBuilder &builder, Value tag, Type type, Value buffer,
              triton::gcu::FirstLastUserAnalysis &userAnalysis,
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
            auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
            for (unsigned i = 0; i < srcType.getRank(); ++i) {
              if (i == axis) {
                offsets.push_back(builder.create<arith::ConstantIntOp>(
                    loc, 0, builder.getI32Type()));
              } else {
                offsets.push_back(builder.create<arith::MulIOp>(
                    loc,
                    builder.create<arith::ConstantIntOp>(loc, numElems[i],
                                                         builder.getI32Type()),
                    builder.create<arith::IndexCastOp>(
                        loc, builder.getI32Type(), warpIds[i])));
              }
            }
            auto output =
                syncAllocOp(builder, loc, op.getOperation(), userAnalysis,
                            replaced2Origin, outputType);
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
                  loc, dst, src, mergedOffsets, defaultValue, tag,
                  ValueRange{zero});
              auto [oriOutputStrides, oriOutputOffset] =
                  outputType.getStridesAndOffset();
              builder.create<memref::ReinterpretCastOp>(
                  loc, outputType, dst, oriOutputOffset,
                  SmallVector<int64_t>(numElems.begin(), numElems.end()),
                  oriOutputStrides);
            } else {
              builder.create<memref_ext::SliceStartOp>(loc, output, buffer,
                                                       offsets, defaultValue,
                                                       tag, ValueRange{zero});
            }
            builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                              totalNumElems);
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
  const GCUArch arch = GCUArch::GCU300;

private:
  void applyReduce(triton::ReduceOp op, OpBuilder &rewriter,
                   ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                   const std::array<int64_t, 3> &reduceDims,
                   int64_t reduceAxis) const {
    if (succeeded(applyVectorizationImpl(op, rewriter, outputs, inputs,
                                         reduceDims, reduceAxis))) {
      return;
    }
    if (succeeded(applySmallSizeImpl(op, rewriter, outputs, inputs, reduceDims,
                                     reduceAxis))) {
      return;
    }
    applyScalarImpl(op, rewriter, outputs, inputs, reduceDims, reduceAxis);
  }

  LogicalResult applyBuiltinImpl(triton::ReduceOp op, OpBuilder &rewriter,
                                 ArrayRef<Value> outputs,
                                 ArrayRef<Value> inputs,
                                 const std::array<int64_t, 3> &reduceDims,
                                 int64_t reduceAxis) const {
    auto loc = op.getLoc();
    auto &opList = op.getCombineOp().front().getOperations();
    if (patternMatch<triton::ExternElementwiseOp, triton::ReduceReturnOp>(
            &opList)) {
      // Reduce Max/Min
      auto externElementwiseOp =
          cast<triton::ExternElementwiseOp>(opList.front());
      auto symbol = externElementwiseOp.getSymbol();
      if ((symbol == "__nv_fmaxf" || symbol == "__nv_max" ||
           symbol == "__nv_umax" || symbol == "__nv_fminf" ||
           symbol == "__nv_min" || symbol == "__nv_umin") &&
          (reduceDims[1] % 16 == 0 && reduceDims[2] % 512 == 0)) {
        Value output = outputs[0];
        Value input = inputs[0];
        Value workspace = nullptr;
        if (reduceAxis == 2) {
          std::array<int64_t, 3> dims = {reduceDims[0], reduceDims[1], 1};
          auto elementTy = cast<MemRefType>(input.getType()).getElementType();
          auto bitWidth = elementTy.getIntOrFloatBitWidth();
          if (bitWidth == 8) {
            dims[2] = 512;
          } else if (bitWidth == 16) {
            dims[2] = 256;
          } else if (bitWidth == 32) {
            dims[2] = 128;
          } else {
            return failure();
          }
          workspace = rewriter.create<memref::AllocOp>(
              loc, MemRefType::get(dims, elementTy));
        } else {
          workspace = output;
        }

        if (symbol == "__nv_fmaxf") {
          rewriter.create<gcu::ReduceOp>(loc, gcu::ReduceOperation::MAXF,
                                         output, input, workspace, reduceAxis);
        } else if (symbol == "__nv_max") {
          rewriter.create<gcu::ReduceOp>(loc, gcu::ReduceOperation::MAXSI,
                                         output, input, workspace, reduceAxis);
        } else if (symbol == "__nv_umax") {
          rewriter.create<gcu::ReduceOp>(loc, gcu::ReduceOperation::MAXUI,
                                         output, input, workspace, reduceAxis);
        } else if (symbol == "__nv_fminf") {
          rewriter.create<gcu::ReduceOp>(loc, gcu::ReduceOperation::MINF,
                                         output, input, workspace, reduceAxis);
        } else if (symbol == "__nv_min") {
          rewriter.create<gcu::ReduceOp>(loc, gcu::ReduceOperation::MINSI,
                                         output, input, workspace, reduceAxis);
        } else {
          rewriter.create<gcu::ReduceOp>(loc, gcu::ReduceOperation::MINUI,
                                         output, input, workspace, reduceAxis);
        }
        if (reduceDims[1] % 16 == 0 && reduceDims[2] % 512 == 0 &&
            reduceAxis == 2) {
          rewriter.create<memref::DeallocOp>(loc, workspace);
        }
      } else {
        return failure();
      }
    } else if ((patternMatch<arith::AddIOp, triton::ReduceReturnOp>(&opList) ||
                patternMatch<arith::AddFOp, triton::ReduceReturnOp>(&opList)) &&
               (reduceDims[2] % 128 == 0 && reduceDims[1] % 128 == 0)) {
      // Reduce Sum
      Value output = outputs[0];
      Value input = inputs[0];
      Value workspace = output;
      rewriter.create<gcu::ReduceOp>(loc, gcu::ReduceOperation::SUM, output,
                                     input, workspace, reduceAxis);
    } else {
      return failure();
    }
    doMemFence(rewriter, op);
    return success();
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
    unsigned minBpe = targetInfo[arch].supportI64 ? 8 : 4;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32() &&
          (targetInfo[arch].supportI64 && !elementType.isInteger(64))) {
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
      elementTypes.push_back(elementType);
    }
    auto numVacc = maxBpe / minBpe;
    if (numVacc > 4) {
      return failure();
    }
    int64_t vectorizeAxis = 3;
    unsigned vectorLength = targetInfo[arch].vaccSizeInBytes / minBpe;
    for (auto i = 2; i >= 0; --i) {
      if (reduceDims[i] >= vectorLength) {
        vectorizeAxis = i;
        break;
      }
    }
    if (vectorizeAxis == 3) {
      return failure();
    }
    while (numVacc < targetInfo[arch].preferVaccNum) {
      int64_t axis = 3;
      unsigned vLen = 2 * vectorLength;
      for (auto i = 2; i >= 0; --i) {
        if (reduceDims[i] >= vLen) {
          axis = i;
          break;
        }
      }
      numVacc *= 2;
      if (axis == 3 || (vectorizeAxis == 2 && axis != 2)) {
        break;
      } else {
        vectorizeAxis = axis;
        vectorLength = vLen;
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
                        return rewriter.create<arith::ConstantIntOp>(
                            loc, dim, rewriter.getI32Type());
                      });
      auto tag = getPrivateDTETag(rewriter, op);
      llvm::transform(inputs, std::back_inserter(tmpBuffers), [&](auto input) {
        auto memrefTy = cast<MemRefType>(input.getType());
        auto elementTy = memrefTy.getElementType();
        auto tmpBuffer = rewriter.create<memref::AllocOp>(
            loc,
            MemRefType::get(ArrayRef<int64_t>{reduceInputDims}, elementTy));
        rewriter.create<memref_ext::TransposeStartOp>(
            loc, tmpBuffer, input, transposeLayoutValue, tag, ValueRange{zero});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag, ValueRange{zero},
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
            loc, MemRefType::get(reduceOutputDims, rewriter.getIntegerType(8)),
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
    llvm::transform(elementTypes, std::back_inserter(vectorTypes),
                    [vectorLength](auto elementTy) {
                      return VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             elementTy);
                    });

    auto vLength = rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
    SmallVector<Value, loopUnrollTime> cur;
    SmallVector<Value, loopUnrollTime> next;
    auto loopLimit = reduceAxis == 1 || !isReduce1D
                         ? reduceInputDims[1]
                         : reduceInputDims[2] / vectorLength;
    auto loopCnt = loopLimit > loopUnrollTime ? loopUnrollTime : loopLimit;
    auto loopCntValue = rewriter.create<arith::ConstantIndexOp>(loc, loopCnt);
    if (reduceAxis == 1) {
      rewriter.create<scf::ForOp>(
          loc, zero,
          rewriter.create<arith::ConstantIndexOp>(loc, reduceInputDims[0]), one,
          ValueRange{},
          [&](OpBuilder &builder, Location loc, Value iter0,
              ValueRange iterArgs) {
            builder.create<scf::ForOp>(
                loc, zero,
                builder.create<arith::ConstantIndexOp>(loc, reduceInputDims[2]),
                vLength, ValueRange{},
                [&](OpBuilder &builder, Location loc, Value iter2,
                    ValueRange iterArgs) {
                  for (unsigned i = 0; i < loopCnt; ++i) {
                    for (unsigned j = 0; j < numOutput; ++j) {
                      cur.emplace_back(builder.create<vector::LoadOp>(
                          loc, vectorTypes[j], inputs[j],
                          ValueRange{
                              iter0,
                              builder.create<arith::ConstantIndexOp>(loc, i),
                              iter2}));
                    }
                  }
                  auto loop = builder.create<scf::ForOp>(
                      loc, loopCntValue,
                      builder.create<arith::ConstantIndexOp>(
                          loc, reduceInputDims[1]),
                      loopCntValue, cur,
                      [&](OpBuilder &builder, Location loc, Value iter1,
                          ValueRange iterArgs) {
                        next.resize(loopCnt * numOutput);
                        for (unsigned i = 0; i < loopCnt; ++i) {
                          for (unsigned j = 0; j < numOutput; ++j) {
                            next[i * numOutput + j] =
                                builder.create<vector::LoadOp>(
                                    loc, vectorTypes[j], inputs[j],
                                    ValueRange{
                                        iter0,
                                        builder.create<arith::AddIOp>(
                                            loc,
                                            builder
                                                .create<arith::ConstantIndexOp>(
                                                    loc, i),
                                            iter1),
                                        iter2});
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
                  cur.assign(loop.getResults().begin(),
                             loop.getResults().end());
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
                  for (unsigned i = 0; i < numOutput; ++i) {
                    builder.create<vector::StoreOp>(
                        loc, *iter++, reduceOutputs[i],
                        ValueRange{iter0, zero, iter2});
                  }
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
    } else {
      // reduceAxis == 2
      rewriter.create<scf::ForOp>(
          loc, zero,
          rewriter.create<arith::ConstantIndexOp>(loc, reduceInputDims[0]), one,
          ValueRange{},
          [&](OpBuilder &builder, Location loc, Value iter0,
              ValueRange iterArgs) {
            if (isReduce1D) {
              builder.create<scf::ForOp>(
                  loc, zero,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         reduceInputDims[1]),
                  one, ValueRange{},
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    for (unsigned i = 0; i < loopCnt; ++i) {
                      for (unsigned j = 0; j < numOutput; ++j) {
                        cur.emplace_back(builder.create<vector::LoadOp>(
                            loc, vectorTypes[j], inputs[j],
                            ValueRange{iter0, iter1,
                                       builder.create<arith::ConstantIndexOp>(
                                           loc, i * vectorLength)}));
                      }
                    }
                    auto loop = builder.create<scf::ForOp>(
                        loc,
                        builder.create<arith::ConstantIndexOp>(
                            loc, loopCnt * vectorLength),
                        builder.create<arith::ConstantIndexOp>(
                            loc, reduceInputDims[2]),
                        builder.create<arith::ConstantIndexOp>(
                            loc, loopCnt * vectorLength),
                        cur,
                        [&](OpBuilder &builder, Location loc, Value iter2,
                            ValueRange iterArgs) {
                          next.resize(loopCnt * numOutput);
                          for (unsigned i = 0; i < loopCnt; ++i) {
                            for (unsigned j = 0; j < numOutput; ++j) {
                              next[i * numOutput + j] =
                                  builder.create<vector::LoadOp>(
                                      loc, vectorTypes[j], inputs[j],
                                      ValueRange{
                                          iter0, iter1,
                                          builder.create<arith::AddIOp>(
                                              loc,
                                              builder.create<
                                                  arith::ConstantIndexOp>(
                                                  loc, i * vectorLength),
                                              iter2)});
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
                                    loc, builder, combineOp, args,
                                    vectorLength));
                          }
                          vectorizeCombineOpTerminator(loc, builder,
                                                       terminatorOperands);
                        });
                    cur.reserve(cur.size() * 2);
                    cur.assign(loop.getResults().begin(),
                               loop.getResults().end());
                    auto iter = cur.begin();
                    while (loopCnt != 1) {
                      loopCnt /= 2;
                      for (unsigned i = 0; i < loopCnt; ++i) {
                        cur.append(vectorizeCombineOpWithoutTerminator(
                            loc, builder, combineOp,
                            ValueRange(iter, 2 * numOutput), vectorLength));
                        iter = std::next(iter, 2 * numOutput);
                      }
                    }
                    auto results =
                        vReduce(loc, builder, combineOp,
                                ValueRange(iter, numOutput), vectorLength);
                    for (unsigned i = 0; i < numOutput; ++i) {
                      if (cast<MemRefType>(reduceOutputs[i].getType())
                              .getElementType()
                              .isInteger(1)) {
                        results[i] = rewriter.create<arith::TruncIOp>(
                            loc, rewriter.getI1Type(), results[i]);
                      }
                      builder.create<memref::StoreOp>(
                          loc, results[i], reduceOutputs[i],
                          ValueRange{iter0, iter1, zero});
                    }
                    builder.create<scf::YieldOp>(loc);
                  });
            } else {
              builder.create<scf::ForOp>(
                  loc, zero,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         reduceInputDims[1]),
                  loopCntValue, ValueRange{},
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    for (unsigned i = 0; i < loopCnt; ++i) {
                      for (unsigned j = 0; j < numOutput; ++j) {
                        cur.emplace_back(builder.create<vector::LoadOp>(
                            loc, vectorTypes[j], inputs[j],
                            ValueRange{
                                iter0,
                                builder.create<arith::AddIOp>(
                                    loc,
                                    builder.create<arith::ConstantIndexOp>(loc,
                                                                           i),
                                    iter1),
                                zero}));
                      }
                    }
                    auto loop = builder.create<scf::ForOp>(
                        loc,
                        builder.create<arith::ConstantIndexOp>(loc,
                                                               vectorLength),
                        builder.create<arith::ConstantIndexOp>(
                            loc, reduceInputDims[2]),
                        builder.create<arith::ConstantIndexOp>(loc,
                                                               vectorLength),
                        cur,
                        [&](OpBuilder &builder, Location loc, Value iter2,
                            ValueRange iterArgs) {
                          next.resize(loopCnt * numOutput);
                          for (unsigned i = 0; i < loopCnt; ++i) {
                            for (unsigned j = 0; j < numOutput; ++j) {
                              next[i * numOutput + j] =
                                  builder.create<vector::LoadOp>(
                                      loc, vectorTypes[j], inputs[j],
                                      ValueRange{
                                          iter0,
                                          builder.create<arith::AddIOp>(
                                              loc,
                                              builder.create<
                                                  arith::ConstantIndexOp>(loc,
                                                                          i),
                                              iter1),
                                          iter2});
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
                                    loc, builder, combineOp, args,
                                    vectorLength));
                          }
                          vectorizeCombineOpTerminator(loc, builder,
                                                       terminatorOperands);
                        });
                    for (unsigned i = 0; i < loopCnt; ++i) {
                      for (unsigned j = 0; j < numOutput; ++j) {
                        auto results =
                            vReduce(loc, builder, combineOp,
                                    ValueRange(loop.getResults().slice(
                                        i * numOutput, numOutput)),
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
                                iter0,
                                builder.create<arith::AddIOp>(
                                    loc,
                                    builder.create<arith::ConstantIndexOp>(loc,
                                                                           i),
                                    iter1),
                                zero});
                      }
                    }
                    builder.create<scf::YieldOp>(loc);
                  });
            }
            builder.create<scf::YieldOp>(loc);
          });
    }

    for (auto buffer : tmpBuffers) {
      rewriter.create<memref::DeallocOp>(loc, buffer);
    }
    if (needTranspose) {
      for (unsigned i = 0; i < numOutput; ++i) {
        auto memrefTy = cast<MemRefType>(outputs[i].getType());
        auto tag = getPrivateDTETag(rewriter, op);
        rewriter.create<memref_ext::TransposeStartOp>(
            loc, outputs[i], reduceOutputs[i], transposeLayoutValue, tag,
            ValueRange{zero});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag, ValueRange{zero},
            rewriter.create<arith::ConstantIndexOp>(loc,
                                                    memrefTy.getNumElements()));
        rewriter.create<memref::DeallocOp>(loc, reduceOutputs[i]);
      }
      doMemFence(rewriter, op);
    } else {
      rewriter.create<gcu::MFenceOp>(loc, gcu::MFenceType::Local);
    }
    return success();
  }

  LogicalResult
  applyVectorizationImplV2(triton::ReduceOp op, OpBuilder &rewriter,
                           ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                           const std::array<int64_t, 3> &reduceDims,
                           int64_t reduceAxis) const {
    SmallVector<Type> elementTypes;
    unsigned maxBpe = 4;
    unsigned minBpe = targetInfo[arch].supportI64 ? 8 : 4;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32() &&
          (targetInfo[arch].supportI64 && !elementType.isInteger(64))) {
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
      elementTypes.push_back(elementType);
    }
    auto numVacc = maxBpe / minBpe;
    if (numVacc > 4) {
      return failure();
    }
    int64_t vectorizeAxis = 3;
    unsigned vectorLength = targetInfo[arch].vaccSizeInBytes / minBpe;
    for (auto i = 2; i >= 0; --i) {
      if (reduceDims[i] >= vectorLength) {
        vectorizeAxis = i;
        break;
      }
    }
    if (vectorizeAxis == 3) {
      return failure();
    }
    while (numVacc < targetInfo[arch].preferVaccNum) {
      int64_t axis = 3;
      unsigned vLen = 2 * vectorLength;
      for (auto i = 2; i >= 0; --i) {
        if (reduceDims[i] >= vLen) {
          axis = i;
          break;
        }
      }
      numVacc *= 2;
      if (axis == 3 || (vectorizeAxis == 2 && axis != 2)) {
        break;
      } else {
        vectorizeAxis = axis;
        vectorLength = vLen;
      }
    }

    auto vectorTypes = llvm::to_vector(
        llvm::map_range(elementTypes, [vectorLength](auto elementTy) {
          return VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
        }));
    auto loc = op.getLoc();
    SmallVector<Value, 4> reduceBuffers;
    llvm::transform(outputs, std::back_inserter(reduceBuffers),
                    [&](auto output) {
                      if (vectorizeAxis == reduceAxis) {
                        auto memrefTy = cast<MemRefType>(output.getType());
                        auto elementTy = memrefTy.getElementType();
                        SmallVector<int64_t, 4> reduceBufferDims(
                            reduceDims.begin(), reduceDims.end());
                        reduceBufferDims[vectorizeAxis] = vectorLength;
                        Value buffer = rewriter.create<memref::AllocOp>(
                            loc, MemRefType::get(reduceBufferDims, elementTy));
                        return buffer;
                      } else {
                        return output;
                      }
                    });
    auto tag = getPrivateDTETag(rewriter, op);
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto stride =
        std::accumulate(reduceDims.begin() + reduceAxis, reduceDims.end(), 1,
                        std::multiplies<unsigned>());
    auto elementsPerStride = stride / reduceDims[reduceAxis];
    auto numElements = product(reduceDims) / reduceDims[reduceAxis];
    if (vectorizeAxis == reduceAxis) {
      elementsPerStride *= vectorLength;
      numElements *= vectorLength;
    }

    for (unsigned i = 0; i < numOutput; ++i) {
      rewriter.create<memref::DmaStartOp>(
          loc, inputs[i], SmallVector<Value, 4>(reduceDims.size(), zero),
          reduceBuffers[i], SmallVector<Value, 4>(reduceDims.size(), zero),
          rewriter.create<arith::ConstantIndexOp>(loc, numElements), tag,
          ValueRange{zero},
          rewriter.create<arith::ConstantIndexOp>(loc, stride),
          rewriter.create<arith::ConstantIndexOp>(loc, elementsPerStride));
      rewriter.create<memref::DmaWaitOp>(
          loc, tag, ValueRange{zero},
          rewriter.create<arith::ConstantIndexOp>(loc, numElements));
    }
    SmallVector<Value, 4> lbs(reduceDims.size(), zero);
    if (vectorizeAxis == reduceAxis) {
      lbs[reduceAxis] =
          rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
    } else {
      lbs[reduceAxis] = one;
    }
    SmallVector<Value, 4> ubs;
    for (auto dim : reduceDims) {
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }

    SmallVector<Value, 4> step(reduceDims.size(), one);
    step[vectorizeAxis] =
        rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
    auto maskType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI1Type());
    Value mask = rewriter.create<vector::ConstantMaskOp>(
        loc, maskType,
        DenseI64ArrayAttr::get(rewriter.getContext(),
                               ArrayRef<int64_t>{vectorLength}));
    unsigned strideOnVectorizeAxis =
        std::accumulate(reduceDims.begin() + vectorizeAxis + 1,
                        reduceDims.end(), 1, std::multiplies<unsigned>());
    if (vectorizeAxis < reduceAxis) {
      strideOnVectorizeAxis /= reduceDims[reduceAxis];
    }

    auto vecIndexTy = VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      rewriter.getIndexType());
    auto vecTy =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI32Type());
    auto indexVec0 = rewriter.create<arith::MulIOp>(
        loc,
        rewriter
            .create<gcu::VectorConvertOp>(
                loc, vecTy,
                rewriter.create<vector::StepOp>(loc, vecIndexTy).getResult())
            .getResult(0),
        rewriter.create<vector::BroadcastOp>(
            loc, vecTy,
            rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32Type(),
                rewriter.getI32IntegerAttr(strideOnVectorizeAxis))));
    Value indexVec1 = indexVec0;
    if (vectorizeAxis < reduceAxis) {
      indexVec1 = rewriter.create<arith::MulIOp>(
          loc, indexVec1,
          rewriter.create<vector::BroadcastOp>(
              loc, vecTy,
              rewriter.create<arith::ConstantOp>(
                  loc, rewriter.getI32Type(),
                  rewriter.getI32IntegerAttr(reduceDims[reduceAxis]))));
    }

    SmallVector<Value, 4> passThruValues;
    for (unsigned i = 0; i < numOutput; ++i) {
      passThruValues.push_back(
          rewriter.create<vector::BroadcastOp>(loc, vectorTypes[i], zero));
    }
    auto &combineOp = op.getCombineOp();
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
                for (unsigned i = 0; i < ivs.size(); ++i) {
                  inputIndices.push_back(ivs[i]);
                  if (i == reduceAxis) {
                    outputIndices.push_back(zero);
                  } else {
                    outputIndices.push_back(inputIndices[i]);
                  }
                }
                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], reduceBuffers[i], outputIndices,
                      indexVec0, mask, passThruValues[i]));
                  resultElemTypes.push_back(vectorTypes[i]);
                }
                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], inputs[i], inputIndices, indexVec1,
                      mask, operands[i]));
                }
                auto executeRegionOp =
                    builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
                executeRegionOp.getRegion().emplaceBlock();
                {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.setInsertionPointToStart(
                      &executeRegionOp.getRegion().back());
                  auto terminatorOperands = vectorizeCombineOpWithoutTerminator(
                      loc, builder, combineOp, operands, vectorLength);
                  vectorizeCombineOpTerminator(loc, builder,
                                               terminatorOperands);
                }
                for (unsigned i = 0; i < numOutput; ++i) {
                  builder.create<vector::ScatterOp>(
                      loc, reduceBuffers[i], outputIndices, indexVec0, mask,
                      executeRegionOp.getResult(i));
                }
              });
          if (reduceAxis == vectorizeAxis) {
            SmallVector<Value, 4> lowerBounds(lbs.begin() + reduceAxis,
                                              lbs.end());
            SmallVector<Value, 4> upperBounds(ubs.begin() + reduceAxis,
                                              ubs.end());
            lowerBounds[0] = zero;
            upperBounds[0] = one;
            scf::buildLoopNest(
                rewriter, loc, lowerBounds, upperBounds,
                ArrayRef<Value>(step.begin() + reduceAxis, step.end()),
                [&](OpBuilder &builder, Location loc, ValueRange innerIters) {
                  auto loopCnt = std::log2(vectorLength);
                  SmallVector<Value, 4> outputIndices;
                  SmallVector<Type, 4> resultElemTypes;
                  for (auto iv : outerIters) {
                    outputIndices.push_back(iv);
                  }
                  for (auto iv : innerIters) {
                    outputIndices.push_back(iv);
                  }
                  builder.create<scf::ForOp>(
                      loc, zero,
                      builder.create<arith::ConstantIndexOp>(loc, loopCnt), one,
                      ValueRange{builder.create<arith::ConstantIndexOp>(
                          loc, vectorLength)},
                      [&](OpBuilder &builder, Location loc, Value iter,
                          ValueRange iterArgs) {
                        SmallVector<Value, 4> args;
                        for (unsigned i = 0; i < numOutput; ++i) {
                          args.push_back(builder.create<vector::GatherOp>(
                              loc, vectorTypes[i], reduceBuffers[i],
                              outputIndices, indexVec0, mask,
                              passThruValues[i]));
                        }
                        auto stride = builder.create<arith::DivSIOp>(
                            loc, iterArgs[0],
                            builder.create<arith::ConstantIndexOp>(loc, 2));
                        SmallVector<Value, 4> indices(outputIndices);
                        indices[vectorizeAxis] = stride;
                        auto strideMask = builder.create<vector::CreateMaskOp>(
                            loc, maskType, ValueRange{stride});
                        for (unsigned i = 0; i < numOutput; ++i) {
                          args.push_back(builder.create<vector::GatherOp>(
                              loc, vectorTypes[i], reduceBuffers[i], indices,
                              indexVec0, strideMask, args[i]));
                          resultElemTypes.push_back(vectorTypes[i]);
                        }
                        auto executeRegion =
                            builder.create<scf::ExecuteRegionOp>(
                                loc, resultElemTypes);
                        executeRegion.getRegion().emplaceBlock();
                        {
                          OpBuilder::InsertionGuard guard(builder);
                          builder.setInsertionPointToStart(
                              &executeRegion.getRegion().back());
                          auto terminatorOperands =
                              vectorizeCombineOpWithoutTerminator(
                                  loc, builder, combineOp, args, vectorLength);
                          vectorizeCombineOpTerminator(loc, builder,
                                                       terminatorOperands);
                        }
                        for (unsigned i = 0; i < numOutput; ++i) {
                          builder.create<vector::ScatterOp>(
                              loc, reduceBuffers[i], outputIndices, indexVec0,
                              strideMask, executeRegion.getResult(i));
                        }
                        builder.create<scf::YieldOp>(loc, ValueRange{stride});
                      });
                  for (unsigned i = 0; i < numOutput; ++i) {
                    builder.create<memref::StoreOp>(
                        loc,
                        builder.create<memref::LoadOp>(loc, reduceBuffers[i],
                                                       outputIndices),
                        outputs[i], outputIndices);
                  }
                });
          }
        });
    if (vectorizeAxis == reduceAxis) {
      for (auto buffer : reduceBuffers)
        rewriter.create<memref::DeallocOp>(loc, buffer);
    }
    doMemFence(rewriter, op);
    return success();
  }

  LogicalResult applySmallSizeImpl(triton::ReduceOp op, OpBuilder &rewriter,
                                   ArrayRef<Value> outputs,
                                   ArrayRef<Value> inputs,
                                   const std::array<int64_t, 3> &reduceDims,
                                   int64_t reduceAxis) const {
    auto loc = op.getLoc();
    if (reduceDims[reduceAxis] == 1) {
      for (auto [input, output] : llvm::zip(inputs, outputs)) {
        rewriter.create<memref::CopyOp>(loc, input, output);
      }
      return success();
    }
    bool isReduce1D = reduceDims[0] == 1 && reduceDims[1] == 1;
    if (isReduce1D) {
      unsigned minBpe = 4;
      unsigned vectorSizeInBytes = targetInfo[arch].vaccSizeInBytes;
      SmallVector<Type> elementTypes;
      for (auto output : outputs) {
        auto elementTy = cast<MemRefType>(output.getType()).getElementType();
        if (!elementTy.isInteger(8) && !elementTy.isInteger(16) &&
            !elementTy.isInteger(32) && !elementTy.isBF16() &&
            !elementTy.isF16() && !elementTy.isF32()) {
          return failure();
        }
        elementTypes.push_back(elementTy);
        auto bpe = mlir::triton::gcu::getBpe(elementTy);
        minBpe = bpe < minBpe ? bpe : minBpe;
      }
      assert(minBpe * reduceDims[2] < vectorSizeInBytes);
      unsigned vectorLength = vectorSizeInBytes / minBpe;

      SmallVector<Value> values;
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      for (auto [input, elementTy] : llvm::zip(inputs, elementTypes)) {
        Value v = rewriter.create<vector::LoadOp>(
            loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
            input, ValueRange{zero, zero, zero});
        if (elementTy.isBF16() || elementTy.isF16()) {
          values.emplace_back(rewriter.create<arith::ExtFOp>(
              loc,
              VectorType::get(ArrayRef<int64_t>{vectorLength},
                              rewriter.getF32Type()),
              v));
        } else if (elementTy.isInteger(1) || elementTy.isInteger(8) ||
                   elementTy.isInteger(16)) {
          values.emplace_back(rewriter.create<arith::ExtSIOp>(
              loc,
              VectorType::get(ArrayRef<int64_t>{vectorLength},
                              rewriter.getI32Type()),
              v));
        } else if (elementTy.isF32() || elementTy.isInteger(32)) {
          values.push_back(v);
        } else {
          llvm_unreachable("unsupported element type");
        }
      }
      auto &combineOp = op.getCombineOp();
      auto vaccLen = targetInfo[arch].vaccSizeInBytes / 4;
      auto argNum = values.size();
      auto vrLen = vaccLen / 2;
      unsigned validLength = reduceDims[2];
      if (vectorLength > vaccLen) {
        auto validVaccNum = ceil<int>(validLength, vrLen);
        int numVacc = vectorLength / vaccLen;
        SmallVector<Value> splitValues(argNum * validVaccNum);
        for (size_t i = 0; i < argNum; ++i) {
          auto elementType =
              cast<VectorType>(values[i].getType()).getElementType();
          SmallVector<Type> resultTypes(
              numVacc,
              VectorType::get(ArrayRef<int64_t>{vaccLen}, elementType));
          auto vectorConvertOp = rewriter.create<gcu::VectorConvertOp>(
              loc, resultTypes, values[i]);
          for (auto j = 0; j < validVaccNum; ++j) {
            splitValues[j * argNum + i] = vectorConvertOp.getResult(j);
          }
        }
        splitValues.reserve(splitValues.size() * 2);
        auto iter = splitValues.begin();
        while (validVaccNum != 1) {
          validVaccNum /= 2;
          for (auto i = 0; i < validVaccNum; ++i) {
            splitValues.append(vectorizeCombineOpWithoutTerminator(
                loc, rewriter, combineOp, ValueRange(iter, 2 * argNum),
                vaccLen));
            iter = std::next(iter, 2 * argNum);
          }
        }
        values.assign(iter, iter + argNum);
        validLength = validLength < vrLen ? validLength : vrLen;
      }
      if (validLength == vrLen) {
        for (size_t i = 0; i < argNum; ++i) {
          values.emplace_back(rewriter.create<gcu::VectorMovSftOp>(
              loc, gcu::VectorMovSftMode::SHFLQW, values[i], 2));
        }
        values = vectorizeCombineOpWithoutTerminator(loc, rewriter, combineOp,
                                                     values, vaccLen);
      } else {
        for (size_t i = 0; i < argNum; ++i) {
          values[i] = rewriter.create<gcu::VectorMovSftOp>(
              loc, gcu::VectorMovSftMode::SHFLQW, values[i], 2);
        }
      }
      if (validLength >= vrLen / 2) {
        for (size_t i = 0; i < argNum; ++i) {
          values.emplace_back(rewriter.create<gcu::VectorMovSftOp>(
              loc, gcu::VectorMovSftMode::SHFLQW, values[i], 1));
        }
        values = vectorizeCombineOpWithoutTerminator(loc, rewriter, combineOp,
                                                     values, vaccLen);
      } else {
        for (size_t i = 0; i < argNum; ++i) {
          values[i] = rewriter.create<gcu::VectorMovSftOp>(
              loc, gcu::VectorMovSftMode::SHFLQW, values[i], 1);
        }
      }
      if (validLength >= vrLen / 4) {
        for (size_t i = 0; i < argNum; ++i) {
          values.emplace_back(rewriter.create<gcu::VectorMovSftOp>(
              loc, gcu::VectorMovSftMode::SHFLB, values[i], 8));
        }
        values = vectorizeCombineOpWithoutTerminator(loc, rewriter, combineOp,
                                                     values, vaccLen);
      } else {
        for (size_t i = 0; i < argNum; ++i) {
          values[i] = rewriter.create<gcu::VectorMovSftOp>(
              loc, gcu::VectorMovSftMode::SHFLB, values[i], 8);
        }
      }
      for (size_t i = 0; i < argNum; ++i) {
        values.emplace_back(rewriter.create<gcu::VectorMovSftOp>(
            loc, gcu::VectorMovSftMode::SHFLB, values[i], 4));
      }
      values = vectorizeCombineOpWithoutTerminator(loc, rewriter, combineOp,
                                                   values, vaccLen);
      for (size_t i = 0; i < argNum; ++i) {
        values[i] = rewriter.create<vector::ExtractElementOp>(
            loc, values[i], rewriter.create<arith::ConstantIndexOp>(loc, 15));
        auto elementType = elementTypes[i];
        if (elementType.isBF16() || elementType.isF16()) {
          values[i] =
              rewriter.create<arith::TruncFOp>(loc, elementType, values[i]);
        } else if (elementType.isInteger(1) || elementType.isInteger(8) ||
                   elementType.isInteger(16)) {
          values[i] =
              rewriter.create<arith::TruncIOp>(loc, elementType, values[i]);
        }

        rewriter.create<memref::StoreOp>(loc, values[i], outputs[i],
                                         ValueRange{zero, zero, zero});
      }
      return success();
    }
    return failure();
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
      if (kind == vector::CombiningKind::MAXNUMF) {
        assert(vecValues.size() == 1);
        SmallVector<Value> values{
            builder.create<vector::ReductionOp>(loc, *kind, vecValues[0])};
        return values;
      }
    }

    SmallVector<Value> values;
    SmallVector<Type> elementTypes;
    for (auto v : vecValues) {
      auto elementTy = cast<VectorType>(v.getType()).getElementType();
      elementTypes.push_back(elementTy);
      if (elementTy.isBF16() || elementTy.isF16()) {
        values.emplace_back(builder.create<arith::ExtFOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getF32Type()),
            v));
      } else if (elementTy.isInteger(1) || elementTy.isInteger(8) ||
                 elementTy.isInteger(16)) {
        values.emplace_back(builder.create<arith::ExtSIOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getI32Type()),
            v));
      } else if (elementTy.isF32() || elementTy.isInteger(32)) {
        values.push_back(v);
      } else {
        llvm_unreachable("unsupported element type");
      }
    }
    auto argNum = values.size();
    auto vaccLen = targetInfo[arch].vaccSizeInBytes / 4;
    if (vectorLength != vaccLen) {
      auto splitNum = vectorLength / vaccLen;
      SmallVector<Value> splitValues(argNum * splitNum);
      for (size_t i = 0; i < argNum; ++i) {
        auto elementType =
            cast<VectorType>(values[i].getType()).getElementType();
        SmallVector<Type> resultTypes(
            splitNum, VectorType::get(ArrayRef<int64_t>{vaccLen}, elementType));
        auto vectorConvertOp =
            builder.create<gcu::VectorConvertOp>(loc, resultTypes, values[i]);
        for (auto j = 0; j < splitNum; ++j) {
          splitValues[j * argNum + i] = vectorConvertOp.getResult(j);
        }
      }
      splitValues.reserve(splitValues.size() * 2);
      auto iter = splitValues.begin();
      while (splitNum != 1) {
        splitNum /= 2;
        for (auto i = 0; i < splitNum; ++i) {
          splitValues.append(vectorizeCombineOpWithoutTerminator(
              loc, builder, combineOp, ValueRange(iter, 2 * argNum), vaccLen));
          iter = std::next(iter, 2 * argNum);
        }
      }
      values.assign(iter, iter + argNum);
    }
    for (size_t i = 0; i < argNum; ++i) {
      values.emplace_back(builder.create<gcu::VectorMovSftOp>(
          loc, gcu::VectorMovSftMode::SHFRQW, values[i], 2));
    }
    values = vectorizeCombineOpWithoutTerminator(loc, builder, combineOp,
                                                 values, vaccLen);
    for (size_t i = 0; i < argNum; ++i) {
      values.emplace_back(builder.create<gcu::VectorMovSftOp>(
          loc, gcu::VectorMovSftMode::SHFRQW, values[i], 1));
    }
    values = vectorizeCombineOpWithoutTerminator(loc, builder, combineOp,
                                                 values, vaccLen);
    for (size_t i = 0; i < argNum; ++i) {
      values.emplace_back(builder.create<gcu::VectorMovSftOp>(
          loc, gcu::VectorMovSftMode::SHFRB, values[i], 8));
    }
    values = vectorizeCombineOpWithoutTerminator(loc, builder, combineOp,
                                                 values, vaccLen);
    for (size_t i = 0; i < argNum; ++i) {
      values.emplace_back(builder.create<gcu::VectorMovSftOp>(
          loc, gcu::VectorMovSftMode::SHFRB, values[i], 4));
    }
    values = vectorizeCombineOpWithoutTerminator(loc, builder, combineOp,
                                                 values, vaccLen);
    for (size_t i = 0; i < argNum; ++i) {
      values.emplace_back(builder.create<vector::BroadcastOp>(
          loc, values[i].getType(),
          builder.create<vector::ExtractElementOp>(
              loc, values[i],
              builder.create<arith::ConstantIndexOp>(loc, 16))));
    }
    values = vectorizeCombineOpWithoutTerminator(loc, builder, combineOp,
                                                 values, vaccLen);
    for (size_t i = 0; i < argNum; ++i) {
      values[i] = builder.create<vector::ExtractElementOp>(
          loc, values[i], builder.create<arith::ConstantIndexOp>(loc, 0));
    }

    for (size_t i = 0; i < argNum; ++i) {
      auto elementType = elementTypes[i];
      if (elementType.isBF16() || elementType.isF16()) {
        values[i] =
            builder.create<arith::TruncFOp>(loc, elementType, values[i]);
      } else if (elementType.isInteger(1) || elementType.isInteger(8) ||
                 elementType.isInteger(16)) {
        values[i] =
            builder.create<arith::TruncIOp>(loc, elementType, values[i]);
      }
    }
    return values;
  }
};
} // namespace

void mlir::triton::populateReduceOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin) {
  patterns.add<TTReduceOpLowering>(converter, patterns.getContext(),
                                   userAnalysis, replaced2Origin);
}
