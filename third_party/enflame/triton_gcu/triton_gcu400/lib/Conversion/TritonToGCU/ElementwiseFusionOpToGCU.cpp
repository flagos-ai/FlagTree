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
#include <queue>
#include <set>
#include <string>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

static const char *const kIsContinual = "IsContinual";

struct GCUElementwiseFusionOpLowering
    : SharedConversionPattern<triton::gcu::ElementwiseFusionRegionOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::ElementwiseFusionRegionOp op,
                  SharedConversionPattern::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op);
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(
        op.getRegion().back().front().getResultTypes().front());

    SmallVector<Value> results;
    SmallVector<Value> outputs;
    DenseSet<Type> elementTypeSet;
    bool needCvtDataLayout = false;
    for (auto [type, opResult] :
         llvm::zip(op.getResultTypes(), op.getResults())) {
      auto resultType =
          dyn_cast<MemRefType>(getTypeConverter()->convertType(type));
      auto lastUser = userAnalysis.getLastUser(opResult);
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
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto operand = adaptor.getOperands()[i];
      auto operandType = operand.getType();
      if (isa<MemRefType>(operandType)) {
        auto totalNumElems =
            triton::gcu::getTotalElemsPerThread(op.getOperandTypes()[i]);
        auto elementTy = cast<MemRefType>(operandType).getElementType();
        elementTypes.push_back(elementTy);
        elementTypeSet.insert(elementTy);
        if (elementTy.isInteger(1)) {
          needCvtDataLayout = true;
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
        elementTypes.push_back(operandType);
        inputs.push_back(operand);
      }
    }

    for (auto &o : op.getRegion().back().without_terminator()) {
      for (auto type : o.getResultTypes()) {
        auto elementTy = getTypeConverter()->convertType(
            cast<TensorType>(type).getElementType());
        if (!elementTy.isInteger(1) || needCvtDataLayout) {
          elementTypeSet.insert(elementTy);
        }
      }
    }

    unsigned maxBpe = 1;
    unsigned minBpe = 8;
    for (auto elementTy : elementTypeSet) {
      auto bpe = mlir::triton::gcu::getBpe(elementTy);
      maxBpe = bpe > maxBpe ? bpe : maxBpe;
      minBpe = bpe < minBpe ? bpe : minBpe;
    }
    unsigned vectorLength = oaccSizeInBytes / minBpe;

    constexpr unsigned loopUnrollTime = 16;
    auto loopLimit = ceil<unsigned>(totalNumElems, vectorLength);
    auto loopCnt = loopUnrollTime > loopLimit ? loopLimit : loopUnrollTime;

    auto insertPoint = rewriter.saveInsertionPoint();
    SmallVector<IRMapping> operandMaps(loopCnt);

    Value mask;
    DenseMap<Operation *, Value> offsets;
    auto useLoadStoreInstrOps =
        trySimplifyLoadStore(op, rewriter, offsets, mask);
    bool disableLoadStroreInstrOptimize = false;
    bool isSmallSize = totalNumElems < vectorLength;

    DenseMap<unsigned, unsigned> broadcastInfo;
    DenseSet<unsigned> broadcastOnDim0;
    for (auto &o : op.getRegion().back().without_terminator()) {
      if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(o)) {
        auto srcType = broadcastOp.getSrc().getType();
        auto resultType = broadcastOp.getType();
        auto rank = srcType.getRank();
        unsigned broadcastAxis = -1;
        for (unsigned i = 0; i < rank; ++i) {
          if (srcType.getDimSize(i) != resultType.getDimSize(i)) {
            broadcastAxis = i;
            break;
          }
        }
        if (auto arg = llvm::dyn_cast<BlockArgument>(broadcastOp.getSrc())) {
          auto argNum =
              dyn_cast<BlockArgument>(broadcastOp.getSrc()).getArgNumber();
          if (broadcastAxis == 0) {
            broadcastOnDim0.insert(argNum);
            auto elemsPerThread = triton::gcu::getElemsPerThread(srcType);
            auto elementNum = std::accumulate(
                elemsPerThread.begin() + broadcastAxis, elemsPerThread.end(), 1,
                std::multiplies<unsigned>());
            auto elementTy =
                dyn_cast<MemRefType>(inputs[argNum].getType()).getElementType();
            if (vectorLength > elementNum) {
              Value v = rewriter.create<vector::LoadOp>(
                  loc,
                  VectorType::get(ArrayRef<int64_t>{elementNum}, elementTy),
                  inputs[argNum],
                  ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0)});
              v = rewriter
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          elementTy),
                          SmallVector<Value>(vectorLength / elementNum, v))
                      .getResult(0);
              for (unsigned j = 0; j < loopCnt; ++j) {
                operandMaps[j].map(op.getRegion().getArgument(argNum), v);
              }
            } else if (vectorLength == elementNum) {
              auto v = rewriter.create<vector::LoadOp>(
                  loc,
                  VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
                  inputs[argNum],
                  ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 0)});
              for (unsigned j = 0; j < loopCnt; ++j) {
                operandMaps[j].map(op.getRegion().getArgument(argNum), v);
              }
            } else {
              if (elementNum > vectorLength * loopCnt) {
                assert(loopCnt == loopUnrollTime);
                vectorLength = elementNum / loopCnt;
              }
              auto cnt = elementNum / vectorLength;
              SmallVector<Value> values(cnt);
              auto vlen =
                  4 * oaccSizeInBytes / mlir::triton::gcu::getBpe(elementTy);
              if (vlen > elementNum) {
                vlen = elementNum;
              }
              auto numVec = vlen / vectorLength;
              for (unsigned j = 0; j < elementNum / vlen; ++j) {
                Value v = rewriter.create<vector::LoadOp>(
                    loc, VectorType::get(ArrayRef<int64_t>{vlen}, elementTy),
                    inputs[argNum],
                    ValueRange{rewriter.create<arith::ConstantIndexOp>(
                        loc, j * vlen)});
                auto convertOp = rewriter.create<gcu::VectorConvertOp>(
                    loc,
                    SmallVector<Type>(
                        numVec, VectorType::get(ArrayRef<int64_t>{vectorLength},
                                                elementTy)),
                    v);
                for (unsigned k = 0; k < numVec; ++k) {
                  values[j * numVec + k] = convertOp.getResult(k);
                }
              }
              for (unsigned j = 0; j < loopCnt; ++j) {
                operandMaps[j].map(op.getRegion().getArgument(argNum),
                                   values[j % cnt]);
              }
            }
          } else if (broadcastAxis == rank - 1) {
            auto elemsPerThread = triton::gcu::getElemsPerThread(resultType);
            auto elementNum = elemsPerThread[broadcastAxis];
            if (elementNum > vectorLength * loopCnt) {
              assert(loopCnt == loopUnrollTime);
              vectorLength = elementNum / loopCnt;
            }
            broadcastInfo[argNum] = elementNum;
          } else {
            llvm_unreachable("unsupported broadcast axis");
          }
        } else {
          llvm_unreachable("unsupported broadcast op");
        }
      }
    }

    triton::gcu::TritonGCUBuilder b(loc, rewriter);
    SmallVector<Value> initValues;

    for (auto i = 0; i < inputs.size(); ++i) {
      if (broadcastOnDim0.contains(i) || broadcastInfo.contains(i)) {
        continue;
      }
      auto type = inputs[i].getType();
      if (isa<MemRefType>(type)) {
        initValues.emplace_back(b.tarAddr(inputs[i]));
      } else if (auto ptrType = dyn_cast<gcu::PtrType>(type)) {
        Value memref = rewriter.create<mlir::gcu::PtrToMemRefOp>(
            loc,
            MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                            ptrType.getElementType()),
            inputs[i]);
        for (unsigned j = 0; j < loopCnt; ++j) {
          operandMaps[j].map(op.getRegion().getArgument(i), memref);
        }
      }
    }

    unsigned cnt = initValues.size();
    for (auto output : outputs) {
      initValues.emplace_back(b.tarAddr(output));
    }

    auto tarStride = b.tarValue(oaccSizeInBytes);

    Value step;
    DenseMap<Operation *, unsigned> map;

    for (auto &o : op.getRegion().back().without_terminator()) {
      if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(o)) {
        auto startIdx = makeRangeOp.getStart();
        auto elementTy = makeRangeOp.getResult().getType().getElementType();
        Value start =
            rewriter.create<arith::ConstantIntOp>(loc, elementTy, startIdx)
                .getResult();
        if (!getSlicedAxies(makeRangeOp.getType()).empty()) {
          start = rewriter.create<arith::AddIOp>(
              loc,
              rewriter.create<arith::MulIOp>(
                  loc,
                  rewriter.create<arith::IndexCastOp>(
                      loc, elementTy,
                      getWarpIds(rewriter, loc, makeRangeOp.getType()).front()),
                  rewriter.create<arith::ConstantIntOp>(loc, elementTy,
                                                        totalNumElems)),
              start);
        }
        map[&o] = initValues.size();
        initValues.emplace_back(
            rewriter
                .create<gcu::VectorStepOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
                    start)
                .getResult());
      }
    }

    if (!isSmallSize) {
      DenseMap<Value, Value> addrMap;
      for (auto [op, offset] : offsets) {
        if (!useLoadStoreInstrOps.contains(op))
          continue;
        auto ptr = op->getOperand(0);
        assert(isa<BlockArgument>(ptr));
        auto ptrType =
            cast<gcu::PtrType>(getTypeConverter()->convertType(ptr.getType()));
        Value addr;
        auto it = addrMap.find(ptr);
        if (it != addrMap.end()) {
          addr = it->second;
        } else {
          addr = b.tarAddr(operandMaps[0].lookup(ptr));
          addrMap[ptr] = addr;
        }
        b.tarJump(
            addr,
            b.tarValue(rewriter.create<arith::MulIOp>(
                loc, offset,
                rewriter.create<arith::ConstantIntOp>(
                    loc, getElementTypeOrSelf(offset),
                    mlir::triton::gcu::getBpe(ptrType.getElementType())))));
        map[op] = initValues.size();
        initValues.emplace_back(addr);
      }
    }

    auto loopBody = [&](OpBuilder &builder, Location loc, Value loopIter,
                        ValueRange iterArgs) {
      SmallVector<Value> args(iterArgs);
      for (unsigned i = 0; i < loopCnt; ++i) {
        for (unsigned j = 0, k = 0; j < inputs.size(); ++j) {
          if (broadcastOnDim0.contains(j)) {
            continue;
          }
          if (broadcastInfo.contains(j)) {
            auto elementNum = broadcastInfo[j];
            auto elementTy =
                cast<MemRefType>(inputs[j].getType()).getElementType();
            auto cnt = loopCnt * vectorLength / elementNum;
            if (elementNum < vectorLength) {
              auto num = vectorLength / elementNum;
              SmallVector<Value> src;
              for (unsigned k = 0; k < num; ++k) {
                src.emplace_back(builder.create<vector::BroadcastOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{elementNum}, elementTy),
                    builder.create<memref::LoadOp>(
                        loc, inputs[j],
                        ValueRange{builder.create<arith::AddIOp>(
                            loc,
                            builder.create<arith::MulIOp>(
                                loc, loopIter,
                                builder.create<arith::ConstantIndexOp>(loc,
                                                                       cnt)),
                            builder.create<arith::ConstantIndexOp>(
                                loc, i * num + k))})));
              }
              auto v = builder
                           .create<gcu::VectorConvertOp>(
                               loc,
                               VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               elementTy),
                               src)
                           .getResult(0);
              operandMaps[i].map(op.getRegion().getArgument(j), v);
            } else {
              auto num = elementNum / vectorLength;
              if (i % num == 0) {
                auto v = builder.create<vector::BroadcastOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
                    builder.create<memref::LoadOp>(
                        loc, inputs[j],
                        ValueRange{builder.create<arith::AddIOp>(
                            loc,
                            builder.create<arith::MulIOp>(
                                loc, loopIter,
                                builder.create<arith::ConstantIndexOp>(loc,
                                                                       cnt)),
                            builder.create<arith::ConstantIndexOp>(loc,
                                                                   i / num))}));
                for (unsigned k = 0; k < num; ++k) {
                  operandMaps[i + k].map(op.getRegion().getArgument(j), v);
                }
              }
            }
          } else if (auto memrefTy =
                         dyn_cast<MemRefType>(inputs[j].getType())) {
            auto elementTy = memrefTy.getElementType();
            Value tarAddr = args[k];
            operandMaps[i].map(op.getRegion().getArgument(j),
                               generateSplitTarLoad(b, builder, elementTy,
                                                    vectorLength, tarAddr,
                                                    tarStride, loc));
            args[k] = tarAddr;
            ++k;
          } else if (isa<gcu::PtrType>(inputs[j].getType())) {
            continue;
          } else {
            operandMaps[i].map(op.getRegion().getArgument(j), inputs[j]);
          }
        }
      }

      auto iterRange = op.getRegion().back().without_terminator();
      auto cur = iterRange.begin();
      while (cur != iterRange.end()) {
        auto &op = *cur;
        if (auto maskedLoadOp = dyn_cast<triton::gcu::MaskedLoadOp>(op)) {
          for (unsigned i = 0; i < loopCnt; ++i) {
            auto result = maskedLoadOp.getResult();
            auto elementTy = result.getType().getElementType();
            auto vecTy =
                VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
            if (useLoadStoreInstrOps.contains(&op) &&
                (!disableLoadStroreInstrOptimize || !maskedLoadOp.getMask())) {
              if (map.contains(&op)) {
                operandMaps[i].map(result,
                                   b.tarLoad(vecTy, args[map[&op]], tarStride));
              } else {
                assert(isSmallSize);
                auto mask = builder.create<vector::ConstantMaskOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{vectorLength},
                                    builder.getIntegerType(1)),
                    DenseI64ArrayAttr::get(builder.getContext(),
                                           ArrayRef<int64_t>{totalNumElems}));
                operandMaps[i].map(
                    result,
                    builder.create<vector::MaskedLoadOp>(
                        loc, vecTy,
                        operandMaps[i].lookup(maskedLoadOp.getPtr()),
                        ValueRange{builder.create<arith::AddIOp>(
                            loc,
                            builder.create<arith::IndexCastOp>(
                                loc, builder.getIndexType(), offsets[&op]),
                            builder.create<arith::AddIOp>(
                                loc,
                                builder.create<arith::ConstantIndexOp>(
                                    loc, i * vectorLength),
                                loopIter))},
                        mask,
                        builder.create<arith::ConstantOp>(
                            loc, DenseElementsAttr::get(
                                     vecTy, builder.getZeroAttr(elementTy)))));
              }
            } else if (offsets.contains(&op)) {
              assert(maskedLoadOp.getMask());
              auto mask = operandMaps[i].lookup(maskedLoadOp.getMask());
              if (isSmallSize) {
                mask = builder.create<arith::AndIOp>(
                    loc,
                    builder.create<vector::ConstantMaskOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        DenseI64ArrayAttr::get(
                            builder.getContext(),
                            ArrayRef<int64_t>{totalNumElems})),
                    mask);
              }
              auto other = maskedLoadOp.getOther();
              operandMaps[i].map(
                  result,
                  builder.create<vector::MaskedLoadOp>(
                      loc, vecTy, operandMaps[i].lookup(maskedLoadOp.getPtr()),
                      ValueRange{builder.create<arith::AddIOp>(
                          loc,
                          builder.create<arith::IndexCastOp>(
                              loc, builder.getIndexType(), offsets[&op]),
                          builder.create<arith::AddIOp>(
                              loc,
                              builder.create<arith::ConstantIndexOp>(
                                  loc, i * vectorLength),
                              loopIter))},
                      mask,
                      other ? operandMaps[i].lookup(other)
                            : builder.create<arith::ConstantOp>(
                                  loc,
                                  DenseElementsAttr::get(
                                      vecTy, builder.getZeroAttr(elementTy)))));
            } else {
              handleMaskedLoadOp(maskedLoadOp, builder, operandMaps[i],
                                 vectorLength, needCvtDataLayout);
            }
          }
          ++cur;
        }
        unsigned i = 0;
        while (i < loopCnt) {
          auto iter = cur;
          while (cur != iterRange.end()) {
            auto &o = *cur;
            if (isa<triton::gcu::MaskedLoadOp>(o)) {
              break;
            } else if (auto cvtLayoutOp =
                           dyn_cast<triton::gpu::ConvertLayoutOp>(o)) {
              operandMaps[i].map(cvtLayoutOp.getResult(),
                                 operandMaps[i].lookup(cvtLayoutOp.getSrc()));
            } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(o)) {
              operandMaps[i].map(broadcastOp.getResult(),
                                 operandMaps[i].lookup(broadcastOp.getSrc()));
            } else if (auto maskedStoreOp =
                           dyn_cast<triton::gcu::MaskedStoreOp>(o)) {
              if (useLoadStoreInstrOps.contains(&o) &&
                  (!disableLoadStroreInstrOptimize ||
                   !maskedStoreOp.getMask())) {
                auto v = operandMaps[i].lookup(maskedStoreOp.getValue());
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
                if (map.contains(&o)) {
                  b.tarStore(v, args[map[&o]], tarStride);
                } else {
                  assert(isSmallSize);
                  auto mask = builder.create<vector::ConstantMaskOp>(
                      loc,
                      VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      builder.getIntegerType(1)),
                      DenseI64ArrayAttr::get(builder.getContext(),
                                             ArrayRef<int64_t>{totalNumElems}));
                  builder.create<vector::MaskedStoreOp>(
                      loc, operandMaps[i].lookup(maskedStoreOp.getPtr()),
                      ValueRange{builder.create<arith::AddIOp>(
                          loc,
                          builder.create<arith::IndexCastOp>(
                              loc, builder.getIndexType(), offsets[&o]),
                          builder.create<arith::AddIOp>(
                              loc,
                              builder.create<arith::ConstantIndexOp>(
                                  loc, i * vectorLength),
                              loopIter))},
                      mask, v);
                }
              } else if (offsets.contains(&o)) {
                assert(maskedStoreOp.getMask());
                auto v = operandMaps[i].lookup(maskedStoreOp.getValue());
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
                auto mask = operandMaps[i].lookup(maskedStoreOp.getMask());
                if (isSmallSize) {
                  mask = builder.create<arith::AndIOp>(
                      loc,
                      builder.create<vector::ConstantMaskOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          builder.getIntegerType(1)),
                          DenseI64ArrayAttr::get(
                              builder.getContext(),
                              ArrayRef<int64_t>{totalNumElems})),
                      mask);
                }
                builder.create<vector::MaskedStoreOp>(
                    loc, operandMaps[i].lookup(maskedStoreOp.getPtr()),
                    ValueRange{builder.create<arith::AddIOp>(
                        loc,
                        builder.create<arith::IndexCastOp>(
                            loc, builder.getIndexType(), offsets[&o]),
                        builder.create<arith::AddIOp>(
                            loc,
                            builder.create<arith::ConstantIndexOp>(
                                loc, i * vectorLength),
                            loopIter))},
                    mask, v);
              } else {
                handleMaskedStoreOp(maskedStoreOp, builder, operandMaps[i],
                                    vectorLength, needCvtDataLayout);
              }
            } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(o)) {
              handleBitcastOp(bitcastOp, builder, operandMaps[i], vectorLength);
            } else if (auto splatOp = dyn_cast<triton::SplatOp>(o)) {
              if (i == 0) {
                OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(insertPoint);
                handleSplatOp(splatOp, builder, operandMaps[i], vectorLength,
                              needCvtDataLayout);
              } else {
                operandMaps[i].map(splatOp.getResult(),
                                   operandMaps[0].lookup(splatOp.getResult()));
              }
            } else if (auto constantOp = dyn_cast<arith::ConstantOp>(o)) {
              if (i == 0) {
                OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(insertPoint);
                handleConstantOp(constantOp, builder, operandMaps[i],
                                 vectorLength, needCvtDataLayout);
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
                OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(insertPoint);
                auto elementTy =
                    makeRangeOp.getResult().getType().getElementType();
                step = builder.create<vector::BroadcastOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
                    builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                         vectorLength));
              }
              operandMaps[i].map(makeRangeOp.getResult(), args[map[&o]]);
              args[map[&o]] =
                  rewriter.create<arith::AddIOp>(loc, args[map[&o]], step);
            } else {
              handleCommonOp(o, builder, operandMaps[i], vectorLength,
                             needCvtDataLayout);
            }
            ++cur;
          }
          if (++i != loopCnt) {
            cur = iter;
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
            Value tarAddr = args[cnt + j];
            generateSplitTarStore(b, builder, v, tarAddr, tarStride, loc);
            args[cnt + j] = tarAddr;
          }
        }
        builder.create<scf::YieldOp>(loc, args);
      }
    };

    if (!useLoadStoreInstrOps.empty()) {
      auto step =
          rewriter.create<arith::ConstantIndexOp>(loc, vectorLength * loopCnt);
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems);
      if (mask) {
        rewriter.create<scf::IfOp>(
            loc, mask,
            [&](OpBuilder &builder, Location loc) {
              builder.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                         initValues, loopBody);
              builder.create<scf::YieldOp>(loc);
            },
            [&](OpBuilder &builder, Location loc) {
              disableLoadStroreInstrOptimize = true;
              rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                          initValues, loopBody);
              builder.create<scf::YieldOp>(loc);
            });
      } else {
        rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                    initValues, loopBody);
      }
    } else {
      rewriter.create<scf::ForOp>(
          loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
          rewriter.create<arith::ConstantIndexOp>(
              loc, ceil<unsigned>(totalNumElems, vectorLength * loopCnt)),
          rewriter.create<arith::ConstantIndexOp>(loc, 1), initValues,
          loopBody);
    }
    leaveTritionOp(rewriter, op);
    if (results.empty()) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, results);
    }
    return success();
  }

private:
  DenseSet<Operation *>
  trySimplifyLoadStore(triton::gcu::ElementwiseFusionRegionOp op,
                       OpBuilder &builder, DenseMap<Operation *, Value> &map,
                       Value &mask) const {
    DenseSet<Value> offsets;
    DenseSet<Value> masks;
    DenseMap<Value, SmallVector<Operation *>> mask2ops;
    DenseSet<Operation *> useLoadStoreInstrOps;
    mask = nullptr;
    for (auto &o : op.getRegion().back()) {
      if (auto maskedLoadOp = dyn_cast<triton::gcu::MaskedLoadOp>(o)) {
        auto attr = o.getAttrOfType<BoolAttr>(kIsContinual);
        if (attr && attr.getValue()) {
          auto offset = maskedLoadOp.getOffset();
          map[&o] = offset;
          offsets.insert(offset);
          if (auto mask = maskedLoadOp.getMask()) {
            masks.insert(mask);
            mask2ops[mask].push_back(&o);
          } else {
            useLoadStoreInstrOps.insert(&o);
          }
        }
      } else if (auto maskedStoreOp = dyn_cast<triton::gcu::MaskedStoreOp>(o)) {
        auto attr = o.getAttrOfType<BoolAttr>(kIsContinual);
        if (attr && attr.getValue()) {
          auto offset = maskedStoreOp.getOffset();
          map[&o] = offset;
          offsets.insert(offset);
          if (auto mask = maskedStoreOp.getMask()) {
            masks.insert(mask);
            mask2ops[mask].push_back(&o);
          } else {
            useLoadStoreInstrOps.insert(&o);
          }
        }
      }
    }

    auto generator =
        [&builder](
            const std::set<Operation *, bool (*)(mlir::Operation *,
                                                 mlir::Operation *)> &opSet,
            IRMapping &mapper) {
          for (auto op : opSet) {
            if (mapper.contains(op->getResult(0))) {
              continue;
            }
            if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
              auto startIdx = makeRangeOp.getStart();
              auto elementTy =
                  makeRangeOp.getResult().getType().getElementType();
              auto loc = op->getLoc();
              Value start =
                  builder.create<arith::ConstantIntOp>(loc, elementTy, startIdx)
                      .getResult();
              if (!getSlicedAxies(makeRangeOp.getType()).empty()) {
                auto totalNumElems = triton::gcu::getTotalElemsPerThread(
                    makeRangeOp.getResult().getType());
                start = builder.create<arith::AddIOp>(
                    loc,
                    builder.create<arith::MulIOp>(
                        loc,
                        builder.create<arith::IndexCastOp>(
                            loc, elementTy,
                            getWarpIds(builder, loc, makeRangeOp.getType())
                                .front()),
                        builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                             totalNumElems)),
                    start);
              }
              mapper.map(makeRangeOp.getResult(), start);
            } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
              mapper.map(splatOp.getResult(),
                         mapper.lookupOrNull(splatOp.getSrc()));
            } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
              auto loc = op->getLoc();
              auto result = constantOp.getResult();
              mapper.map(result,
                         builder.create<arith::ConstantOp>(
                             loc, getElementTypeOrSelf(result),
                             dyn_cast<DenseElementsAttr>(constantOp.getValue())
                                 .getSplatValue<TypedAttr>()));
            } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
              auto loc = op->getLoc();
              auto result = cmpOp.getResult();
              auto lhs = mapper.lookupOrNull(cmpOp.getLhs());
              auto rhs = mapper.lookupOrNull(cmpOp.getRhs());
              auto predicate = cmpOp.getPredicate();
              if (predicate == arith::CmpIPredicate::slt ||
                  predicate == arith::CmpIPredicate::ult) {
                auto size = builder.create<arith::SubIOp>(loc, rhs, lhs);
                auto totalNumElems = triton::gcu::getTotalElemsPerThread(
                    cmpOp.getResult().getType());
                mapper.map(result, builder.create<arith::CmpIOp>(
                                       loc, arith::CmpIPredicate::sge, size,
                                       builder.create<arith::ConstantIntOp>(
                                           loc, getElementTypeOrSelf(size),
                                           totalNumElems)));
              } else {
                mapper.map(result, builder.create<arith::CmpIOp>(loc, predicate,
                                                                 lhs, rhs));
              }
            } else {
              auto cloneOp = builder.clone(*op, mapper);
              for (auto [result, newResult] :
                   llvm::zip(op->getResults(), cloneOp->getResults())) {
                newResult.setType(getElementTypeOrSelf(result));
                mapper.map(result, newResult);
              }
            }
          }
        };

    std::set<Operation *, bool (*)(Operation *, Operation *)> opSet(
        [](Operation *lhs, Operation *rhs) {
          return lhs->isBeforeInBlock(rhs);
        });
    std::queue<Operation *> workList;
    IRMapping mapper;
    for (auto mask : masks) {
      if (auto defOp = mask.getDefiningOp()) {
        workList.push(defOp);
        opSet.insert(defOp);
        while (!workList.empty()) {
          auto o = workList.front();
          workList.pop();
          for (auto operand : o->getOperands()) {
            auto defOp = operand.getDefiningOp();
            if (defOp) {
              if (isa<arith::AndIOp, arith::AddIOp, arith::ExtSIOp,
                      arith::ExtUIOp, arith::ConstantOp, triton::MakeRangeOp,
                      triton::SplatOp>(defOp)) {
                workList.push(defOp);
                opSet.insert(defOp);
              } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
                auto predicate = cmpOp.getPredicate();
                if ((predicate == arith::CmpIPredicate::slt ||
                     predicate == arith::CmpIPredicate::ult ||
                     predicate == arith::CmpIPredicate::sge ||
                     predicate == arith::CmpIPredicate::uge) &&
                    isa_and_nonnull<arith::ConstantOp, triton::SplatOp>(
                        cmpOp.getRhs().getDefiningOp())) {
                  workList.push(defOp);
                  opSet.insert(defOp);
                } else {
                  mask2ops.erase(mask);
                  opSet.clear();
                  continue;
                }
              } else {
                mask2ops.erase(mask);
                opSet.clear();
                continue;
              }
            } else {
              if (!isa<IntegerType>(operand.getType())) {
                mask2ops.erase(mask);
                opSet.clear();
                continue;
              }
              mapper.map(
                  operand,
                  op.getOperand(cast<BlockArgument>(operand).getArgNumber()));
            }
          }
        }
        generator(opSet, mapper);
        opSet.clear();
      }
    }

    for (auto offset : offsets) {
      auto defOp = offset.getDefiningOp();
      assert(defOp);
      workList.push(defOp);
      opSet.insert(defOp);
      while (!workList.empty()) {
        auto o = workList.front();
        workList.pop();
        for (auto operand : o->getOperands()) {
          auto defOp = operand.getDefiningOp();
          if (defOp) {
            workList.push(defOp);
            opSet.insert(defOp);
          } else {
            assert(isa<IntegerType>(operand.getType()));
            assert(isa<BlockArgument>(operand));
            mapper.map(
                operand,
                op.getOperand(cast<BlockArgument>(operand).getArgNumber()));
          }
        }
      }
      generator(opSet, mapper);
      opSet.clear();
    }
    for (auto [k, v] : map) {
      map[k] = mapper.lookup(v);
    }
    for (auto const &[k, ops] : mask2ops) {
      if (mask) {
        mask = builder.create<arith::AndIOp>(op.getLoc(), mask,
                                             mapper.lookupOrNull(k));
      } else {
        mask = mapper.lookupOrNull(k);
      }
      for (auto op : ops) {
        useLoadStoreInstrOps.insert(op);
      }
    }
    return useLoadStoreInstrOps;
  }

  unsigned getSplitLength(unsigned splitIndex, unsigned vectorLength,
                          unsigned maxVectorLength) const {
    unsigned remainingLength = vectorLength - splitIndex * maxVectorLength;
    return std::min(remainingLength, maxVectorLength);
  }

  Value convertLoadStoreMask(OpBuilder &builder, Value mask,
                             unsigned vectorLength, Location loc) const {
    if (!mask) {
      return mask;
    }

    auto defOp = mask.getDefiningOp();
    auto maskElementType = getElementTypeOrSelf(mask.getType());
    if (isa<arith::AndIOp, arith::XOrIOp, arith::OrIOp>(defOp)) {
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(defOp);
      auto lhs = builder
                     .create<gcu::VectorConvertOp>(
                         loc,
                         VectorType::get(ArrayRef<int64_t>{vectorLength},
                                         builder.getIntegerType(8)),
                         defOp->getOperand(0))
                     .getResult(0);
      auto rhs = builder
                     .create<gcu::VectorConvertOp>(
                         loc,
                         VectorType::get(ArrayRef<int64_t>{vectorLength},
                                         builder.getIntegerType(8)),
                         defOp->getOperand(1))
                     .getResult(0);

      Operation *clonedOp = builder.clone(*defOp);
      clonedOp->setOperand(0, lhs);
      clonedOp->setOperand(1, rhs);
      clonedOp->getResult(0).setType(VectorType::get(
          ArrayRef<int64_t>{vectorLength}, builder.getIntegerType(8)));
      builder.restoreInsertionPoint(ip);
      mask = builder
                 .create<gcu::VectorConvertOp>(
                     loc,
                     VectorType::get(ArrayRef<int64_t>{vectorLength},
                                     maskElementType),
                     clonedOp->getResult(0))
                 .getResult(0);
    }
    return mask;
  }

  Value generateSplitTarLoad(triton::gcu::TritonGCUBuilder &b,
                             OpBuilder &builder, Type elementTy,
                             unsigned vectorLength, Value &tarAddr,
                             const Value &tarStride, Location loc) const {
    unsigned bpe = mlir::triton::gcu::getBpe(elementTy);
    unsigned maxVectorLength = 4 * oaccSizeInBytes / bpe;

    if (vectorLength <= maxVectorLength) {
      return b.tarLoad(
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy), tarAddr,
          tarStride);
    }

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;
    SmallVector<Value> loadValues;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      Value v =
          b.tarLoad(VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy),
                    tarAddr, tarStride);
      loadValues.push_back(v);
    }

    SmallVector<Type> resultTypes;
    resultTypes.push_back(
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy));
    return builder.create<gcu::VectorConvertOp>(loc, resultTypes, loadValues)
        .getResult(0);
  }

  void generateSplitTarStore(triton::gcu::TritonGCUBuilder &b,
                             OpBuilder &builder, Value v, Value &tarAddr,
                             const Value &tarStride, Location loc) const {
    VectorType vectorType = cast<VectorType>(v.getType());
    Type elementTy = vectorType.getElementType();
    unsigned vectorLength = vectorType.getNumElements();

    unsigned bpe = mlir::triton::gcu::getBpe(elementTy);
    unsigned maxVectorLength = 4 * oaccSizeInBytes / bpe;

    if (vectorLength <= maxVectorLength) {
      return b.tarStore(v, tarAddr, tarStride);
    }

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;
    SmallVector<Type> resultTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      resultTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy));
    }

    auto convertOp = builder.create<gcu::VectorConvertOp>(loc, resultTypes, v);

    for (unsigned i = 0; i < numSplits; ++i) {
      b.tarStore(convertOp.getResult(i), tarAddr, tarStride);
    }
    return;
  }

  void generateSplitMaskedLoadOp(triton::gcu::MaskedLoadOp op, Value mask,
                                 OpBuilder &builder, IRMapping &map,
                                 unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto elementTy = op.getResult().getType().getElementType();
    auto vectorType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
    auto other = op.getOther();
    auto offset = map.lookup(op.getOffset());
    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * oaccSizeInBytes / offsetBpe;

    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (numElements < vectorLength) {
        Value constantMask = builder.create<vector::ConstantMaskOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getIntegerType(1)),
            DenseI64ArrayAttr::get(builder.getContext(),
                                   ArrayRef<int64_t>{numElements}));
        constantMask = builder
                           .create<gcu::VectorConvertOp>(
                               op.getLoc(),
                               VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               builder.getIntegerType(8)),
                               constantMask)
                           .getResult(0);
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(8)),
                       mask)
                   .getResult(0);
        mask = builder.create<arith::AndIOp>(loc, mask, constantMask);
      }
    }

    mask = builder
               .create<gcu::VectorConvertOp>(
                   op.getLoc(),
                   VectorType::get(ArrayRef<int64_t>{vectorLength},
                                   builder.getIntegerType(8)),
                   mask)
               .getResult(0);
    auto elemBitWidth = elementTy.getIntOrFloatBitWidth();
    mask = builder.create<arith::ExtSIOp>(
        loc,
        VectorType::get(ArrayRef<int64_t>{vectorLength},
                        builder.getIntegerType(elemBitWidth)),
        mask);

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> offsetTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      offsetTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, offsetElementType));
    }
    auto offsets =
        builder.create<gcu::VectorConvertOp>(loc, offsetTypes, offset);

    SmallVector<Type> maskTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      maskTypes.push_back(VectorType::get(
          ArrayRef<int64_t>{currentLen}, builder.getIntegerType(elemBitWidth)));
    }
    auto masks = builder.create<gcu::VectorConvertOp>(loc, maskTypes, mask);

    SmallVector<Value> passThruValues;
    if (other) {
      Value otherVec = map.lookup(other);
      SmallVector<Type> passThruTypes;
      for (unsigned i = 0; i < numSplits; ++i) {
        unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
        passThruTypes.push_back(
            VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy));
      }
      auto passThruSplitOp =
          builder.create<gcu::VectorConvertOp>(loc, passThruTypes, otherVec);
      for (unsigned i = 0; i < numSplits; ++i) {
        passThruValues.push_back(passThruSplitOp.getResult(i));
      }
    } else {
      Value zeroScalar =
          triton::gcu::createConstantZero(builder, loc, elementTy);
      for (unsigned i = 0; i < numSplits; ++i) {
        unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
        auto splitVectorType =
            VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy);
        passThruValues.push_back(builder.create<vector::BroadcastOp>(
            loc, splitVectorType, zeroScalar));
      }
    }

    SmallVector<Value> loadValues;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      auto splitVectorType =
          VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy);

      mask = builder
                 .create<gcu::VectorConvertOp>(
                     op.getLoc(),
                     VectorType::get(ArrayRef<int64_t>{currentLen},
                                     builder.getIntegerType(1)),
                     masks.getResult(i))
                 .getResult(0);

      Value gatherResult = builder.create<vector::GatherOp>(
          loc, splitVectorType, map.lookup(op.getPtr()), ValueRange{zero},
          offsets.getResult(i), mask, passThruValues[i]);
      loadValues.push_back(gatherResult);
    }

    auto mergedResult =
        builder.create<gcu::VectorConvertOp>(loc, vectorType, loadValues)
            .getResult(0);
    map.map(op.getResult(), mergedResult);
  }

  void generateSplitMaskedStoreOp(triton::gcu::MaskedStoreOp op, Value mask,
                                  OpBuilder &builder, IRMapping &map,
                                  unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto offset = map.lookup(op.getOffset());
    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());

    auto v = map.lookup(op.getValue());
    auto valueElementType = getElementTypeOrSelf(v.getType());
    auto elemBitWidth = valueElementType.getIntOrFloatBitWidth();

    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * oaccSizeInBytes / offsetBpe;

    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (numElements < vectorLength) {
        Value constantMask = builder.create<vector::ConstantMaskOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getIntegerType(1)),
            DenseI64ArrayAttr::get(builder.getContext(),
                                   ArrayRef<int64_t>{numElements}));
        constantMask = builder
                           .create<gcu::VectorConvertOp>(
                               op.getLoc(),
                               VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               builder.getIntegerType(8)),
                               constantMask)
                           .getResult(0);
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(8)),
                       mask)
                   .getResult(0);
        mask = builder.create<arith::AndIOp>(loc, mask, constantMask);
      }
    }

    mask = builder
               .create<gcu::VectorConvertOp>(
                   op.getLoc(),
                   VectorType::get(ArrayRef<int64_t>{vectorLength},
                                   builder.getIntegerType(8)),
                   mask)
               .getResult(0);
    mask = builder.create<arith::ExtSIOp>(
        loc,
        VectorType::get(ArrayRef<int64_t>{vectorLength},
                        builder.getIntegerType(elemBitWidth)),
        mask);

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> offsetTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      offsetTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, offsetElementType));
    }
    auto offsets =
        builder.create<gcu::VectorConvertOp>(loc, offsetTypes, offset);

    SmallVector<Type> maskTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      maskTypes.push_back(VectorType::get(
          ArrayRef<int64_t>{currentLen}, builder.getIntegerType(elemBitWidth)));
    }
    auto masks = builder.create<gcu::VectorConvertOp>(loc, maskTypes, mask);

    SmallVector<Type> valueTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      valueTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, valueElementType));
    }
    auto values = builder.create<gcu::VectorConvertOp>(loc, valueTypes, v);

    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      mask = builder
                 .create<gcu::VectorConvertOp>(
                     op.getLoc(),
                     VectorType::get(ArrayRef<int64_t>{currentLen},
                                     builder.getIntegerType(1)),
                     masks.getResult(i))
                 .getResult(0);
      builder.create<vector::ScatterOp>(loc, map.lookup(op.getPtr()),
                                        ValueRange{zero}, offsets.getResult(i),
                                        mask, values.getResult(i));
    }
    return;
  }

  Operation *generateSplitSelectOp(arith::SelectOp op, OpBuilder &builder,
                                   IRMapping &map, unsigned vectorLength,
                                   unsigned maxVectorLength) const {
    auto loc = op.getLoc();

    Value condition = map.lookup(op.getCondition());
    Value trueValue = map.lookup(op.getTrueValue());
    Value falseValue = map.lookup(op.getFalseValue());

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> conditionTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      conditionTypes.push_back(VectorType::get(ArrayRef<int64_t>{currentLen},
                                               builder.getIntegerType(1)));
    }
    auto conditionSplits =
        builder.create<gcu::VectorConvertOp>(loc, conditionTypes, condition);

    SmallVector<Type> valueTypes;
    auto valueElementType = getElementTypeOrSelf(op.getResult().getType());
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      valueTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, valueElementType));
    }
    auto trueSplits =
        builder.create<gcu::VectorConvertOp>(loc, valueTypes, trueValue);
    auto falseSplits =
        builder.create<gcu::VectorConvertOp>(loc, valueTypes, falseValue);

    SmallVector<Value> resultSplits;
    for (unsigned i = 0; i < numSplits; ++i) {
      Value resultSplit = builder.create<arith::SelectOp>(
          loc, conditionSplits.getResult(i), trueSplits.getResult(i),
          falseSplits.getResult(i));
      resultSplits.push_back(resultSplit);
    }

    auto mergedResult = builder.create<gcu::VectorConvertOp>(
        loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, valueElementType),
        resultSplits);
    map.map(op.getResult(), mergedResult.getResult(0));
    return mergedResult;
  }

  Operation *generateSplitExtUIOp(arith::ExtUIOp op, OpBuilder &builder,
                                  IRMapping &map, unsigned vectorLength,
                                  unsigned maxVectorLength) const {
    auto loc = op.getLoc();

    Value inValue = map.lookup(op.getIn());
    if (getElementTypeOrSelf(inValue.getType()).isInteger(8)) {
      inValue = builder
                    .create<gcu::VectorConvertOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        inValue)
                    .getResult(0);
    }

    inValue = builder.create<arith::ExtSIOp>(
        loc,
        VectorType::get(ArrayRef<int64_t>{vectorLength},
                        builder.getIntegerType(8)),
        inValue);

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> inTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      inTypes.push_back(VectorType::get(ArrayRef<int64_t>{currentLen},
                                        builder.getIntegerType(1)));
    }
    auto inSplits = builder.create<gcu::VectorConvertOp>(loc, inTypes, inValue);

    SmallVector<Value> resultSplits;
    auto outElementType = getElementTypeOrSelf(op.getOut().getType());
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      Value resultSplit = builder.create<arith::ExtUIOp>(
          loc, VectorType::get(ArrayRef<int64_t>{currentLen}, outElementType),
          inSplits.getResult(i));
      resultSplits.push_back(resultSplit);
    }

    auto mergedResult = builder.create<gcu::VectorConvertOp>(
        loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, outElementType),
        resultSplits);
    map.map(op.getResult(), mergedResult.getResult(0));
    return mergedResult;
  }

  void handleMaskedLoadOp(triton::gcu::MaskedLoadOp op, OpBuilder &builder,
                          IRMapping &map, unsigned vectorLength,
                          bool needCvtDataLayout) const {
    auto loc = op.getLoc();
    auto offset = map.lookup(op.getOffset());
    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    auto elementTy = op.getResult().getType().getElementType();
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * oaccSizeInBytes / offsetBpe;

    auto vectorType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
    auto other = op.getOther();
    auto mask = map.lookupOrNull(op.getMask());

    if (vectorLength > maxVectorLength) {
      mask = convertLoadStoreMask(builder, mask, vectorLength, loc);
      // tcle support load/store for i8/f8 with v8 oacc, so no need to split.
      if (mlir::triton::gcu::getBpe(elementTy) > 1) {
        return generateSplitMaskedLoadOp(op, mask, builder, map, vectorLength);
      }
    }

    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());
    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (needCvtDataLayout) {
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(1)),
                       mask)
                   .getResult(0);
      }
      if (numElements < vectorLength) {
        mask = builder.create<arith::AndIOp>(
            loc, mask,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{numElements})));
      }
    }
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    map.map(op.getResult(),
            builder.create<vector::GatherOp>(
                loc, vectorType, map.lookup(op.getPtr()), ValueRange{zero},
                map.lookup(op.getOffset()), mask,
                other ? builder.create<vector::BroadcastOp>(loc, vectorType,
                                                            map.lookup(other))
                      : builder.create<vector::BroadcastOp>(
                            loc, vectorType,
                            triton::gcu::createConstantZero(builder, loc,
                                                            elementTy))));
  }

  void handleMaskedStoreOp(triton::gcu::MaskedStoreOp op, OpBuilder &builder,
                           IRMapping &map, unsigned vectorLength,
                           bool needCvtDataLayout) const {
    auto v = map.lookup(op.getValue());
    auto offset = map.lookup(op.getOffset());
    auto valueElementType = getElementTypeOrSelf(v.getType());
    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * oaccSizeInBytes / offsetBpe;

    auto loc = op.getLoc();
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto mask = map.lookupOrNull(op.getMask());

    if (vectorLength > maxVectorLength) {
      mask = convertLoadStoreMask(builder, mask, vectorLength, loc);
      // tcle support load/store for i8/f8 with v8 oacc, so no need to split.
      if (mlir::triton::gcu::getBpe(valueElementType) > 1) {
        return generateSplitMaskedStoreOp(op, mask, builder, map, vectorLength);
      }
    }

    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());
    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (needCvtDataLayout) {
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(1)),
                       mask)
                   .getResult(0);
      }
      if (numElements < vectorLength) {
        mask = builder.create<arith::AndIOp>(
            loc, mask,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{numElements})));
      }
    }
    if (dyn_cast<VectorType>(v.getType()).getElementType().isInteger(1)) {
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
    builder.create<vector::ScatterOp>(loc, map.lookup(op.getPtr()),
                                      ValueRange{zero},
                                      map.lookup(op.getOffset()), mask, v);
  }

  void handleConstantOp(arith::ConstantOp op, OpBuilder &builder,
                        IRMapping &map, unsigned vectorLength,
                        bool needCvtDataLayout) const {
    auto loc = op.getLoc();
    auto elementTy = cast<TensorType>(op.getType()).getElementType();
    auto vectorType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
    Value v;
    if (elementTy.isInteger(1)) {
      if (needCvtDataLayout) {
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
        if (dyn_cast<DenseElementsAttr>(op.getValue())
                .getSplatValue<APInt>()
                .isZero()) {
          v = builder.create<vector::ConstantMaskOp>(
              loc, vectorType,
              DenseI64ArrayAttr::get(builder.getContext(),
                                     ArrayRef<int64_t>{0}));
        } else {
          v = builder.create<vector::ConstantMaskOp>(
              loc, vectorType,
              DenseI64ArrayAttr::get(builder.getContext(),
                                     ArrayRef<int64_t>{vectorLength}));
        }
      }
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
                     unsigned vectorLength, bool needCvtDataLayout) const {
    auto loc = op.getLoc();
    auto elementTy = getTypeConverter()->convertType(
        dyn_cast<TensorType>(op.getType()).getElementType());
    Value v;
    if (elementTy.isInteger(1)) {
      if (needCvtDataLayout) {
        v = builder.create<vector::BroadcastOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getIntegerType(8)),
            builder.create<arith::ExtUIOp>(loc, builder.getIntegerType(8),
                                           map.lookup(op.getSrc())));
      } else {
        auto vectorType =
            VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
        auto ifOp = builder.create<scf::IfOp>(
            loc, map.lookup(op.getSrc()),
            [&](OpBuilder &b, Location loc) {
              Value allTrue = b.create<vector::ConstantMaskOp>(
                  loc, vectorType,
                  DenseI64ArrayAttr::get(b.getContext(),
                                         ArrayRef<int64_t>{vectorLength}));
              b.create<scf::YieldOp>(loc, allTrue);
            },
            [&](OpBuilder &b, Location loc) {
              Value allFalse = b.create<vector::ConstantMaskOp>(
                  loc, vectorType,
                  DenseI64ArrayAttr::get(b.getContext(), ArrayRef<int64_t>{0}));
              b.create<scf::YieldOp>(loc, allFalse);
            });
        v = ifOp.getResult(0);
      }
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
    static const llvm::StringSet<> symbolSet = {
        ///  unary op
        "__nv_abs",
        "__nv_llabs",
        "__nv_fabsf",
        "__nv_brev",
        "__nv_brevll",
        "__nv_clz",
        "__nv_clzll"
        "__nv_ffs",
        "__nv_ffsll",
        "__nv_popc",
        "__nv_popcll",
        "__nv_acosf",
        "__nv_acoshf",
        "__nv_asinf",
        "__nv_asinhf",
        "__nv_atanf",
        "__nv_atanhf",
        "__nv_cbrtf",
        "__nv_ceilf",
        "__nv_cosf",
        "__nv_coshf",
        "__nv_cospif",
        "__nv_erfcf",
        "__nv_erfcinvf",
        "__nv_erfcxf",
        "__nv_erff",
        "__nv_erfinvf",
        "__nv_exp10f",
        "__nv_exp2f",
        "__nv_expf",
        "__nv_expm1f",
        "__nv_floorf",
        "__nv_frcp_rd",
        "__nv_frcp_rn",
        "__nv_frcp_ru",
        "__nv_frcp_rz",
        "__nv_frsqrt_rn",
        "__nv_fsqrt_rd",
        "__nv_fsqrt_rn",
        "__nv_fsqrt_ru",
        "__nv_fsqrt_rz",
        "__nv_j0f",
        "__nv_j1f",
        "__nv_lgammaf",
        "__nv_log10f",
        "__nv_log1pf",
        "__nv_log2f",
        "__nv_logbf",
        "__nv_logf",
        "__nv_nearbyintf",
        "__nv_normcdff",
        "__nv_normcdfinvf",
        "__nv_rcbrtf",
        "__nv_rintf",
        "__nv_roundf",
        "__nv_rsqrtf",
        "__nv_saturatef",
        "__nv_sinf",
        "__nv_sinhf",
        "__nv_sinpif",
        "__nv_sqrtf",
        "__nv_tanf",
        "__nv_tanhf",
        "__nv_tgammaf",
        "__nv_truncf",
        "__nv_y0f",
        "__nv_y1f",
        ///  unary int32 float
        "__nv_finitef",
        "__nv_ilogbf",
        "__nv_isinff",
        "__nv_isnanf",
        "__nv_signbitf",
        ///  unary int64 float
        "__nv_llrintf",
        "__nv_llroundf",
        ///  unary type convert
        "__nv_float_as_int",
        "__nv_int_as_float",
        "__nv_float2int_rd",
        "__nv_float2int_rn",
        "__nv_float2int_ru",
        "__nv_float2int_rz",
        "__nv_float2uint_rd",
        "__nv_float2uint_rn",
        "__nv_float2uint_ru",
        "__nv_float2uint_rz",
        "__nv_int2float_rd",
        "__nv_int2float_rn",
        "__nv_int2float_ru",
        "__nv_int2float_rz",
        "__nv_uint2float_rd",
        "__nv_uint2float_rn",
        "__nv_uint2float_ru",
        "__nv_uint2float_rz",
        "__nv_float2ll_rd",
        "__nv_float2ll_rn",
        "__nv_float2ll_ru",
        "__nv_float2ll_rz",
        "__nv_float2ull_rd",
        "__nv_float2ull_rn",
        "__nv_float2ull_ru",
        "__nv_float2ull_rz",
        "__nv_ll2float_rd",
        "__nv_ll2float_rn",
        "__nv_ll2float_ru",
        "__nv_ll2float_rz",
        "__nv_ull2float_rd",
        "__nv_ull2float_rn",
        "__nv_ull2float_ru",
        "__nv_ull2float_rz",
        ///  binary op
        "__nv_hadd",
        "__nv_uhadd",
        "__nv_llmax",
        "__nv_ullmax",
        "__nv_max",
        "__nv_umax",
        "__nv_fmaxf",
        "__nv_llmin",
        "__nv_ullmin",
        "__nv_min",
        "__nv_umin",
        "__nv_fminf",
        "__nv_mul24",
        "__nv_umul24",
        "__nv_mulhi",
        "__nv_umulhi",
        "__nv_mul64hi",
        "__nv_umul64hi",
        "__nv_rhadd",
        "__nv_urhadd",
        "__nv_atan2f",
        "__nv_copysignf",
        "__nv_fadd_rd",
        "__nv_fadd_rn",
        "__nv_fadd_ru",
        "__nv_fadd_rz",
        "__nv_fdimf",
        "__nv_fdiv_rd",
        "__nv_fdiv_rn",
        "__nv_fdiv_ru",
        "__nv_fdiv_rz",
        "__nv_fmodf",
        "__nv_fmul_rd",
        "__nv_fmul_rn",
        "__nv_fmul_ru",
        "__nv_fmul_rz",
        "__nv_fsub_rd",
        "__nv_fsub_rn",
        "__nv_fsub_ru",
        "__nv_fsub_rz",
        "__nv_hypotf",
        "__nv_nextafterf",
        "__nv_powif",
        "__nv_powf",
        "__nv_remainderf",
        ///  binary float float int32
        "__nv_ldexpf",
        "__nv_scalbnf",
        ///  ternary op
        "__nv_fmaf",
        "__nv_fmaf_rd",
        "__nv_fmaf_rn",
        "__nv_fmaf_ru",
        "__nv_fmaf_rz",
    };

    static const std::string mixedPrecisionSymbolPrefixList[] = {
        "__gcu_wadd", "__gcu_add",     "__gcu_wmul",
        "__gcu_mul",  "__gcu_mac",     "__gcu_mas",
        "__gcu_imas", "__gcu_sigmoid", "__gcu_softplus"};

    std::string efSymbol = "__ef_v";
    auto resultTy = VectorType::get(ArrayRef<int64_t>{vectorLength},
                                    getElementTypeOrSelf(op.getResult()));
    if (symbolSet.contains(symbol)) {
      efSymbol += symbol.drop_front(strlen("__nv_"));
      newOp = builder.create<gcu::ExternElementwiseOp>(loc, resultTy, operands,
                                                       efSymbol);
    } else if (symbol == "__nv_ffs") {
      newOp = builder.create<math::CtPopOp>(loc, operands.front().getType(),
                                            operands);
    } else if (llvm::any_of(mixedPrecisionSymbolPrefixList,
                            [&symbol](StringRef symbolPrefix) {
                              return symbol.starts_with(symbolPrefix);
                            })) {
      newOp = builder.create<gcu::ExternElementwiseOp>(loc, resultTy, operands,
                                                       symbol);
    } else {
      llvm_unreachable(
          ("unsupported extern elementwise: " + symbol).str().c_str());
    }
    map.map(op.getResult(), newOp->getResult(0));
  }

  void handleCommonOp(Operation &op, OpBuilder &builder, IRMapping &map,
                      unsigned vectorLength, bool needCvtDataLayout) const {
    Operation *newOp;
    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      auto condition = selectOp.getCondition();
      auto mapValue = map.lookup(condition);
      if (getElementTypeOrSelf(mapValue.getType()).isInteger(8)) {
        map.map(condition,
                builder
                    .create<gcu::VectorConvertOp>(
                        op.getLoc(),
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        mapValue)
                    .getResult(0));
      }

      unsigned maxVectorLength = vectorLength;
      auto conditionDefOp = condition.getDefiningOp();
      if (conditionDefOp && isa<arith::CmpIOp, arith::CmpFOp>(conditionDefOp)) {
        auto lhsValue = map.lookup(conditionDefOp->getOperand(0));
        auto rhsValue = map.lookup(conditionDefOp->getOperand(1));
        unsigned lhsBpe =
            mlir::triton::gcu::getBpe(getElementTypeOrSelf(lhsValue.getType()));
        unsigned rhsBpe =
            mlir::triton::gcu::getBpe(getElementTypeOrSelf(rhsValue.getType()));
        unsigned bpe = std::max(lhsBpe, rhsBpe);
        maxVectorLength = 4 * oaccSizeInBytes / bpe;
      }
      if (vectorLength > maxVectorLength) {
        newOp = generateSplitSelectOp(selectOp, builder, map, vectorLength,
                                      maxVectorLength);
      } else {
        newOp = builder.clone(op, map);
        map.map(condition, mapValue);
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
        auto outElementType = getElementTypeOrSelf(cvtOp.getOut().getType());
        unsigned outBpe = mlir::triton::gcu::getBpe(outElementType);
        unsigned maxVectorLength = 4 * oaccSizeInBytes / outBpe;
        if (vectorLength > maxVectorLength &&
            getElementTypeOrSelf(cvtOp.getIn().getType()).isInteger(1)) {
          newOp = generateSplitExtUIOp(cvtOp, builder, map, vectorLength,
                                       maxVectorLength);
        } else {
          newOp = builder.clone(op, map);
        }
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
      if (isa<arith::CmpFOp, arith::CmpIOp>(op) && needCvtDataLayout) {
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
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<GCUElementwiseFusionOpLowering>(converter, patterns.getContext(),
                                               userAnalysis, replaced2Origin,
                                               pTagPool);
}
