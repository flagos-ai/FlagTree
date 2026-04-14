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
#include <utility>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {
struct TTScanOpLowering : SharedConversionPattern<triton::ScanOp> {
  using SharedConversionPattern::SharedConversionPattern;

  void applyScan(triton::ScanOp op, OpBuilder &rewriter,
                 ArrayRef<Value> outputs, ArrayRef<Value> inputs, Type type,
                 bool reverse) const {
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
                                   reverse))) {
      return;
    }
    return applyScanFallback(op, rewriter, outs, scanInOutDims, scanAxis,
                             reverse);
  }

  LogicalResult applyGeneralScan(triton::ScanOp op, OpBuilder &rewriter,
                                 ArrayRef<Value> outputs,
                                 const std::array<int64_t, 3> &scanInOutDims,
                                 int64_t scanAxis, bool reverse) const {
    auto loc = op.getLoc();
    int64_t vectorizeAxis;
    if (scanAxis == 2) {
      assert(scanInOutDims[0] == 1);
      vectorizeAxis = 1;
    } else {
      assert(scanAxis == 1);
      vectorizeAxis = scanInOutDims[0] > scanInOutDims[2] ? 0 : 2;
    }

    unsigned maxBpe = 4;
    unsigned minBpe = 4;
    for (auto output : outputs) {
      auto elementType = cast<MemRefType>(output.getType()).getElementType();
      if (!elementType.isInteger(1) && !elementType.isInteger(8) &&
          !elementType.isInteger(16) && !elementType.isInteger(32) &&
          !elementType.isBF16() && !elementType.isF16() &&
          !elementType.isF32() && !elementType.isInteger(64)) {
        return failure();
      }
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }

    auto numOacc = maxBpe / minBpe;
    if (numOacc > 4) {
      return failure();
    }

    unsigned vectorLength = oaccSizeInBytes / minBpe;
    if (scanInOutDims[vectorizeAxis] < vectorLength) {
      return failure();
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
            .create<gcu::VectorStepOp>(
                loc, vecTy, rewriter.create<arith::ConstantIntOp>(loc, 0, 32))
            .getResult(),
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
    return success();
  }

  void applyScanFallback(triton::ScanOp op, OpBuilder &rewriter,
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

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto inputType = dyn_cast<TensorType>(op.getSrcs()[0].getType());

    auto slicedAxies = getSlicedAxies(inputType);
    bool isScanDimSplit = slicedAxies.count(op.getAxis());

    auto numInput = op.getSrcs().size();
    auto numOutput = op.getResults().size();

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // create outputs
    SmallVector<Value, 4> outputs;
    SmallVector<Type, 4> outputElemTypes;
    SmallVector<std::pair<Operation *, int>, 4> lastUsers;
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
    }
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
    if (isScanDimSplit || !isValidBlockEncoding) {
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

      // computing in thread 0
      auto isThread0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
      rewriter.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            applyScan(op, builder, mergedOutputs, mergedInputs, mergedInputType,
                      op.getReverse());
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
      applyScan(op, rewriter, outputs,
                SmallVector<Value, 4>(adaptor.getSrcs().begin(),
                                      adaptor.getSrcs().end()),
                inputType, op.getReverse());
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
};
} // namespace

void mlir::triton::populateScanOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<TTScanOpLowering>(converter, patterns.getContext(), userAnalysis,
                                 replaced2Origin, pTagPool);
}
