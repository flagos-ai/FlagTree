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
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
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
#include "Utils/TritonVersionCompat.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

static int64_t getConstantSplatInt(Value val) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(val.getDefiningOp())) {
    Attribute valueAttr = constOp.getValue();

    if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
      if (denseAttr.isSplat()) {
        return denseAttr.getSplatValue<APInt>().getSExtValue();
      }
    }

    if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
      return intAttr.getInt();
    }
  }

  return -1;
}

// Walk the offset definition chain of a masked load/store op and require
// every arith::RemSIOp's M to be a multiple of `granularity`.  When M %
// granularity == 0, remsi never wraps inside any granularity-sized window,
// so element 0 of the offset vector is a valid scalar base for a contiguous
// load/store of `granularity` elements.  Returns false if the op lacks
// IsContinual or any remsi in the chain violates the constraint.
static bool checkRemSiNoWrap(Operation *op, int64_t granularity) {
  auto attr = op->getAttrOfType<BoolAttr>(kIsContinual);
  if (!attr || !attr.getValue())
    return false;

  Value offset;
  if (auto loadOp = dyn_cast<triton::gcu::MaskedLoadOp>(op))
    offset = loadOp.getOffset();
  else if (auto storeOp = dyn_cast<triton::gcu::MaskedStoreOp>(op))
    offset = storeOp.getOffset();
  else
    return false;

  std::queue<Value> workList;
  DenseSet<Value> visited;
  workList.push(offset);
  while (!workList.empty()) {
    auto val = workList.front();
    workList.pop();
    if (!visited.insert(val).second)
      continue;
    auto defOp = val.getDefiningOp();
    if (!defOp)
      continue;
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
      int64_t M = getConstantSplatInt(remOp.getRhs());
      if (M <= 0 || M % granularity != 0)
        return false;
      workList.push(remOp.getLhs());
    } else {
      for (auto operand : defOp->getOperands())
        workList.push(operand);
    }
  }
  return true;
}

// Per-warp contiguous: remsi never wraps inside the whole warp block
// (totalNumElems elements).  Strongest guarantee - enables tar load/store
// and the offset-scalarization fast path.
static bool isOffsetContiguousInBlock(Operation *op) {
  Value offset;
  if (auto loadOp = dyn_cast<triton::gcu::MaskedLoadOp>(op))
    offset = loadOp.getOffset();
  else if (auto storeOp = dyn_cast<triton::gcu::MaskedStoreOp>(op))
    offset = storeOp.getOffset();
  else
    return false;
  int64_t totalNumElems = triton::gcu::getTotalElemsPerThread(offset.getType());
  return checkRemSiNoWrap(op, totalNumElems);
}

// Per-OACC-vector contiguous: remsi never wraps inside a single OACC vector
// (vectorLength elements), but may wrap across vectors.  Weaker than
// per-warp - the per-vector base (element 0 of the offset vector) is still a
// valid contiguous load/store base, and the outer loop's args[] accumulation
// advances the base across vectors.  Requires M % vectorLength == 0 for
// every remsi in the offset chain.
static bool isOffsetContiguousPerVector(Operation *op, unsigned vectorLength) {
  return checkRemSiNoWrap(op, static_cast<int64_t>(vectorLength));
}

static bool canUseAllocaForVboolValue(Value val) {
  SetVector<Value> worklist;
  worklist.insert(val);
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value cur = worklist[i];
    for (auto *user : cur.getUsers()) {
      if (auto elementwiseFusionRegionOp =
              dyn_cast<triton::gcu::ElementwiseFusionRegionOp>(user)) {
        if (llvm::any_of(
                elementwiseFusionRegionOp.getOperands(), [&](auto operand) {
                  auto type = dyn_cast<TensorType>(operand.getType());
                  if (type && type.getElementType().isInteger(1)) {
                    auto defOp = operand.getDefiningOp();
                    while (isa<triton::ExpandDimsOp>(defOp) &&
                           !triton::gcu::isExpensiveView(
                               cast<triton::ExpandDimsOp>(defOp))) {
                      defOp = defOp->getOperand(0).getDefiningOp();
                    }
                    return !isa<triton::gcu::ElementwiseFusionRegionOp>(defOp);
                  }
                  return false;
                })) {
          return false;
        }
        if (llvm::any_of(*elementwiseFusionRegionOp.getBody(), [&](auto &op) {
              auto resultTypes = op.getResultTypes();
              if (llvm::any_of(resultTypes, [&](auto resultType) {
                    return cast<TensorType>(resultType)
                        .getElementType()
                        .isInteger(1);
                  })) {
                return !isa<arith::AndIOp>(&op) && !isa<arith::OrIOp>(&op) &&
                       !isa<arith::XOrIOp>(&op) && !isa<arith::CmpIOp>(&op) &&
                       !isa<arith::CmpFOp>(&op) &&
                       !isa<arith::ConstantOp>(&op) &&
                       !isa<triton::BroadcastOp>(&op);
              }
              return false;
            })) {
          return false;
        }
        continue;
      }
      if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(user)) {
        if (triton::gcu::isExpensiveView(expandDimsOp))
          return false;
        worklist.insert(expandDimsOp.getResult());
        continue;
      }
      return false;
    }
  }
  return true;
}

static bool canUseAllocaForValue(Value val) {
  SetVector<Value> worklist;
  worklist.insert(val);
  for (unsigned i = 0; i < worklist.size(); ++i) {
    Value cur = worklist[i];
    for (auto *user : cur.getUsers()) {
      if (isa<triton::gcu::ElementwiseFusionRegionOp>(user))
        continue;
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(user)) {
        if (reduceOp->getNumOperands() != 1) {
          return false;
        }
        auto axis = reduceOp.getAxis();
        auto tensorType = reduceOp.getInputTypes()[0];
        if (tensorType.getRank() - 1 != axis ||
            triton::gcu::getElemsPerThread(tensorType)[axis] *
                    triton::gcu::getBpe(tensorType.getElementType()) <
                kOaccSizeInBytes) {
          return false;
        }
        continue;
      }
      if (auto gatherOp = dyn_cast<triton::GatherOp>(user)) {
        if (gatherOp.getSrc() != cur)
          continue;
        return false;
      }
      if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(user)) {
        auto srcNumElems =
            triton::gcu::getElemsPerThread(expandDimsOp.getSrc().getType());
        auto dstNumElems =
            triton::gcu::getElemsPerThread(expandDimsOp.getType());
        srcNumElems.insert(srcNumElems.begin() + expandDimsOp.getAxis(), 1);
        if (srcNumElems != dstNumElems)
          return false;
        worklist.insert(expandDimsOp.getResult());
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        auto initArgs = forOp.getInitArgs();
        for (unsigned i = 0; i < initArgs.size(); ++i) {
          if (initArgs[i] == cur) {
            auto bodyArg = forOp.getRegionIterArg(i);
            worklist.insert(bodyArg);
            worklist.insert(forOp.getResult(i));
          }
        }
        continue;
      }
      return false;
    }
  }
  return true;
}

struct FusionRegionInfo {
  unsigned totalNumElems;
  unsigned vectorLength;
  bool needCvtDataLayout = false;
  unsigned loopCnt;
  bool unrollFull = false;
  SmallVector<bool> useAlloca;
  SmallVector<bool> useAllocaStore; // store stack data to local memory
  static FusionRegionInfo analyze(
      triton::gcu::ElementwiseFusionRegionOp op,
      SharedConversionPattern<triton::gcu::ElementwiseFusionRegionOp>::OpAdaptor
          adaptor) {
    FusionRegionInfo info;
    info.totalNumElems = triton::gcu::getTotalElemsPerThread(
        op.getBody()->front().getResultTypes().front());

    DenseSet<Type> elementTypeSet;
    for (auto [idx, operand] : llvm::enumerate(adaptor.getOperands())) {
      if (auto type = dyn_cast<MemRefType>(operand.getType())) {
        auto elementTy = type.getElementType();
        if (elementTy.isInteger(1)) {
          auto defOp = operand.getDefiningOp();
          while (isa_and_nonnull<memref::ReinterpretCastOp>(defOp)) {
            defOp = defOp->getOperand(0).getDefiningOp();
          }
          if (isa_and_nonnull<memref::AllocaOp>(defOp)) {
            continue;
          }
          if (llvm::all_of(op.getBody()->getArgument(idx).getUsers(),
                           [](auto user) {
                             return isa_and_nonnull<triton::BroadcastOp>(user);
                           })) {
            continue;
          }
          elementTypeSet.insert(elementTy);
          info.needCvtDataLayout = true;
        } else {
          elementTypeSet.insert(elementTy);
        }
      }
    }
    bool allI1TensorResultUseAlloca =
        !info.needCvtDataLayout &&
        llvm::any_of(op.getResults(), [](auto result) {
          return cast<TensorType>(result.getType())
              .getElementType()
              .isInteger(1);
        });
    for (auto result : op.getResults()) {
      auto elementTy = cast<TensorType>(result.getType()).getElementType();
      if (elementTy.isInteger(1) && allI1TensorResultUseAlloca) {
        if (!canUseAllocaForVboolValue(result)) {
          allI1TensorResultUseAlloca = false;
          elementTypeSet.insert(elementTy);
        }
      } else {
        elementTypeSet.insert(elementTy);
      }
    }
    for (auto &o : op.getRegion().back().without_terminator()) {
      for (auto type : o.getResultTypes()) {
        auto elementTy = dyn_cast<TensorType>(type).getElementType();
        if (!elementTy.isInteger(1)) {
          elementTypeSet.insert(elementTy);
        }
      }
    }
    unsigned maxBpe = 1;
    unsigned minBpe = 8;
    for (auto elementTy : elementTypeSet) {
      auto bpe = mlir::triton::gcu::getBpe(elementTy);
      maxBpe = std::max(maxBpe, bpe);
      minBpe = std::min(minBpe, bpe);
    }
    info.vectorLength = kOaccSizeInBytes / minBpe;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto *defOp = op.getOperand(i).getDefiningOp();
      if (!defOp)
        continue;
      if (auto dotOp = dyn_cast<triton::DotOp>(defOp)) {
        auto accStoreAttr = dotOp->getAttrOfType<StringAttr>(kAccStore);
        if (accStoreAttr && accStoreAttr.getValue() == kAccStoreNone) {
          info.unrollFull = true;
        }
      }
    }

    auto tripCount = ceil<unsigned>(info.totalNumElems, info.vectorLength);
    info.useAllocaStore.assign(op.getNumResults(), false);
    if (info.totalNumElems < info.vectorLength ||
        tripCount > kLoopUnrollTimes) {
      if (allI1TensorResultUseAlloca) {
        minBpe = std::min(minBpe, 1u);
        info.vectorLength = kOaccSizeInBytes / minBpe;
      }
      info.useAlloca.assign(op.getNumResults(), false);
    } else {
      for (auto result : op.getResults()) {
        auto elementTy = cast<TensorType>(result.getType()).getElementType();
        if (elementTy.isInteger(1)) {
          if (!allI1TensorResultUseAlloca)
            info.useAlloca.push_back(false);
          else
            info.useAlloca.push_back(true);
          continue;
        } else {
          info.useAlloca.push_back(canUseAllocaForValue(result));
        }
      }
    }
    if (auto inplaceAttr =
            op->getAttrOfType<IntegerAttr>(kAccReuseInplaceResult)) {
      int64_t inplaceResultdx = inplaceAttr.getInt();
      if (inplaceResultdx >= 0) {
        info.useAlloca[inplaceResultdx] = true;
        bool hasStoreLocal = true;
        if (auto accStore = op->getAttr(kAccStore))
          hasStoreLocal =
              cast<StringAttr>(accStore).getValue() == kAccStoreLocal;
        info.useAllocaStore[inplaceResultdx] =
            hasStoreLocal &&
            !canUseAllocaForValue(op.getResult(inplaceResultdx));
      }
    }

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      if (info.useAlloca[i]) {
        info.unrollFull = true;
        break;
      }
    }

    info.loopCnt = std::min(tripCount, kLoopUnrollTimes);
    return info;
  }
};

static Value narrowToStoreVector(OpBuilder &builder, Location loc, Value v) {
  auto vecTy = cast<VectorType>(v.getType());
  auto elementTy = vecTy.getElementType();
  unsigned maxVectorLength =
      4 * kOaccSizeInBytes / mlir::triton::gcu::getBpe(elementTy);
  unsigned numElems = vecTy.getNumElements();
  if (numElems <= maxVectorLength)
    return v;
  unsigned numParts = numElems / maxVectorLength;
  SmallVector<Type> partTypes(
      numParts, VectorType::get(ArrayRef<int64_t>{maxVectorLength},
                                vecTy.getElementType()));
  return builder.create<gcu::VectorConvertOp>(loc, partTypes, v).getResult(0);
}

// If |v| is an i1 vector, convert it to i8 (the element type expected by
// vector store / tar store).  The conversion is inserted right after the
// defining op of |v| to keep the def-use order valid.  Returns |v| unchanged
// when no conversion is needed.
static Value convertI1ToI8VectorForStore(OpBuilder &builder, Location loc,
                                         Value v, unsigned vectorLength) {
  if (!cast<VectorType>(v.getType()).getElementType().isInteger(1))
    return v;
  OpBuilder::InsertionGuard guard(builder);
  auto defOp = v.getDefiningOp();
  assert(defOp);
  builder.setInsertionPointAfter(defOp);
  return builder
      .create<gcu::VectorConvertOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(8)),
          v)
      .getResult(0);
}

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

    Block::iterator reduceOpIter = op.getBody()->begin();
    while (reduceOpIter != op.getBody()->end()) {
      if (isa<triton::ReduceOp>(*reduceOpIter)) {
        break;
      }
      ++reduceOpIter;
    }

    if (reduceOpIter != op.getBody()->end()) {
      mlir::triton::gcu::ReduceGenerator reduceGenerator(
          cast<triton::ReduceOp>(*reduceOpIter), op.getBody()->getArguments(),
          llvm::make_range(op.getBody()->begin(), reduceOpIter));
      auto results =
          llvm::to_vector(llvm::map_range(op.getResults(), [&](auto result) {
            auto memrefType = dyn_cast<MemRefType>(
                getTypeConverter()->convertType(result.getType()));
            auto lastUser = userAnalysis.getLastUser(result);
            return syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                               replaced2Origin, memrefType);
          }));
      auto inputs =
          reduceGenerator.normalizeInputs(rewriter, loc, adaptor.getOperands());
      auto outputs = reduceGenerator.normalizeOutputs(rewriter, loc, results);
      reduceGenerator.applyReduce(rewriter, loc, outputs, inputs);
      rewriter.replaceOp(op, results);
      return success();
    }

    auto fusionRegionInfo = FusionRegionInfo::analyze(op, adaptor);
    auto totalNumElems = fusionRegionInfo.totalNumElems;
    auto loopCnt = fusionRegionInfo.loopCnt;
    auto vectorLength = fusionRegionInfo.vectorLength;
    auto needCvtDataLayout = fusionRegionInfo.needCvtDataLayout;
    bool isSmallSize = totalNumElems < vectorLength;

    // When AnnotateDotFusion marks this fusion for OACC accumulator reuse, the
    // (single) result is written in place into the tagged operand buffer (the
    // loop iter-arg / OACC), so no separate output buffer is allocated.
    int64_t inplaceOperandIdx = -1;
    if (auto inplaceAttr =
            op->getAttrOfType<IntegerAttr>(kAccReuseInplaceOperand))
      inplaceOperandIdx = inplaceAttr.getInt();

    int64_t inplaceResultIdx = -1;
    if (auto inplaceAttr =
            op->getAttrOfType<IntegerAttr>(kAccReuseInplaceResult))
      inplaceResultIdx = inplaceAttr.getInt();

    SmallVector<Value> results;
    SmallVector<Value> outputs;
    for (auto [i, pair] : llvm::enumerate(
             llvm::zip_equal(op.getResults(), fusionRegionInfo.useAlloca))) {
      auto [result, useAlloca] = pair;
      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(result.getType()));
      auto elementTy = resultType.getElementType();
      if (inplaceResultIdx == static_cast<int64_t>(i)) {
        results.push_back(adaptor.getOperands()[inplaceOperandIdx]);
      } else if (useAlloca) {
        auto allocaOp = rewriter.create<memref::AllocaOp>(loc, resultType);
        allocaOp.setAlignment(kOaccSizeInBytes);
        results.push_back(allocaOp.getResult());
      } else {
        auto lastUser = userAnalysis.getLastUser(result);
        results.push_back(syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                      replaced2Origin, resultType));
      }
      if (elementTy.isInteger(1) && !useAlloca) {
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
                    results.back())),
            0, ArrayRef<int64_t>{totalNumElems}, ArrayRef<int64_t>{1}));
      } else {
        outputs.emplace_back(rewriter.create<memref::ReinterpretCastOp>(
            loc, MemRefType::get(ArrayRef<int64_t>{totalNumElems}, elementTy),
            results.back(), 0, ArrayRef<int64_t>{totalNumElems},
            ArrayRef<int64_t>{1}));
      }
    }

    SmallVector<Value> inputs;
    SmallVector<bool> isAllocaInput(op.getNumOperands(), false);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto operand = adaptor.getOperands()[i];
      auto operandType = operand.getType();
      if (auto memrefTy = dyn_cast<MemRefType>(operandType)) {
        auto elementTy = cast<MemRefType>(operandType).getElementType();
        auto totalNumElems =
            triton::gcu::getTotalElemsPerThread(op.getOperandTypes()[i]);
        if (mlir::triton::gcu::isAllocaInputValue(operand) ||
            i == static_cast<unsigned>(inplaceOperandIdx)) {
          inputs.emplace_back(rewriter.create<memref::ReinterpretCastOp>(
              loc, MemRefType::get(ArrayRef<int64_t>{totalNumElems}, elementTy),
              operand, 0, ArrayRef<int64_t>{totalNumElems},
              ArrayRef<int64_t>{1}));
          fusionRegionInfo.unrollFull = true;
          isAllocaInput[i] = true;
        } else {
          if (elementTy.isInteger(1) &&
              (llvm::any_of(
                  op.getBody()->getArgument(i).getUsers(), [](auto user) {
                    return !isa_and_nonnull<triton::BroadcastOp>(user);
                  }))) {
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
                        mlir::gcu::PtrType::get(rewriter.getContext(),
                                                elementTy),
                        operand)),
                0, ArrayRef<int64_t>{totalNumElems}, ArrayRef<int64_t>{1}));
          } else {
            inputs.emplace_back(rewriter.create<memref::ReinterpretCastOp>(
                loc,
                MemRefType::get(ArrayRef<int64_t>{totalNumElems}, elementTy),
                operand, 0, ArrayRef<int64_t>{totalNumElems},
                ArrayRef<int64_t>{1}));
          }
        }
      } else {
        inputs.push_back(operand);
      }
    }

    auto insertPoint = rewriter.saveInsertionPoint();
    SmallVector<IRMapping> operandMaps(loopCnt);

    Value mask;
    llvm::MapVector<Operation *, Value> offsets;
    DenseSet<Operation *> perVecContiguousOps;
    auto useLoadStoreInstrOps = trySimplifyLoadStore(
        op, rewriter, offsets, mask, vectorLength, perVecContiguousOps);
    bool disableLoadStroreInstrOptimize = false;

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
          auto argNum = arg.getArgNumber();
          if (broadcastAxis == 0) {
            broadcastOnDim0.insert(argNum);
            auto elemsPerThread = triton::gcu::getElemsPerThread(srcType);
            auto elementNum = std::accumulate(
                elemsPerThread.begin() + broadcastAxis, elemsPerThread.end(),
                1u, std::multiplies<unsigned>());
            auto elementTy =
                dyn_cast<MemRefType>(inputs[argNum].getType()).getElementType();
            std::optional<unsigned> vectorLen = std::nullopt;
            Value remappedValue =
                rewriter.getRemappedValue(op->getOperand(argNum));
            for (auto user : llvm::make_filter_range(
                     remappedValue.getUsers(), [](Operation *op) {
                       return isa<memref::ReinterpretCastOp>(op);
                     })) {
              if (llvm::any_of(user->getUsers(), [&](auto op) {
                    if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
                      vectorLen = storeOp.getVectorType().getDimSize(0);
                      return true;
                    }
                    return false;
                  })) {
                break;
              }
            }
            auto emitLoad = [&](Value input, int offset) -> Value {
              auto vlen = vectorLen ? *vectorLen : vectorLength;
              if (vectorLength > vlen) {
                auto len =
                    vectorLength > elementNum ? elementNum : vectorLength;
                auto vectorType =
                    VectorType::get(ArrayRef<int64_t>{vlen}, elementTy);
                SmallVector<Value> values;
                for (unsigned i = 0; i < len / vlen; ++i) {
                  values.emplace_back(rewriter.create<vector::LoadOp>(
                      loc, vectorType, input,
                      ValueRange{rewriter.create<arith::ConstantIndexOp>(
                          loc, offset + i * vlen)}));
                }
                return rewriter
                    .create<gcu::VectorConvertOp>(
                        loc, VectorType::get(ArrayRef<int64_t>{len}, elementTy),
                        values)
                    ->getResult(0);
              } else {
                auto vectorType = VectorType::get(
                    ArrayRef<int64_t>{vectorLength > elementNum ? elementNum
                                                                : vlen},
                    elementTy);
                return rewriter.create<vector::LoadOp>(
                    loc, vectorType, input,
                    ValueRange{
                        rewriter.create<arith::ConstantIndexOp>(loc, offset)});
              }
            };
            if (vectorLength > elementNum) {
              auto v = emitLoad(inputs[argNum], 0);
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
              auto v = emitLoad(inputs[argNum], 0);
              for (unsigned j = 0; j < loopCnt; ++j) {
                operandMaps[j].map(op.getRegion().getArgument(argNum), v);
              }
            } else {
              if (elementNum > vectorLength * loopCnt) {
                assert(loopCnt == kLoopUnrollTimes);
                vectorLength = elementNum / loopCnt;
              }
              auto vlen = vectorLen ? *vectorLen : vectorLength;
              auto tripCount = elementNum / vectorLength;
              if (vlen <= vectorLength) {
                for (unsigned i = 0; i < tripCount; ++i) {
                  auto v = emitLoad(inputs[argNum], i * vectorLength);
                  for (unsigned j = 0; j < loopCnt / tripCount; ++j) {
                    operandMaps[j * tripCount + i].map(
                        op.getRegion().getArgument(argNum), v);
                  }
                }
              } else {
                auto numVec = vlen / vectorLength;
                for (unsigned i = 0; i < elementNum / vlen; ++i) {
                  auto convertOp = rewriter.create<gcu::VectorConvertOp>(
                      loc,
                      SmallVector<Type>(
                          numVec,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          elementTy)),
                      emitLoad(inputs[argNum], i * vlen));
                  for (unsigned j = 0; j < numVec; ++j) {
                    for (unsigned k = 0; k < loopCnt / tripCount; ++k) {
                      operandMaps[k * tripCount + i * numVec + j].map(
                          op.getRegion().getArgument(argNum),
                          convertOp.getResult(j));
                    }
                  }
                }
              }
            }
          } else if (broadcastAxis == rank - 1) {
            auto elemsPerThread = triton::gcu::getElemsPerThread(resultType);
            auto elementNum = elemsPerThread[broadcastAxis];
            if (elementNum > vectorLength * loopCnt) {
              assert(loopCnt == kLoopUnrollTimes);
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

    for (size_t i = 0; i < inputs.size(); ++i) {
      if (broadcastOnDim0.contains(i) || broadcastInfo.contains(i)) {
        continue;
      }
      if (isAllocaInput[i]) {
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
    for (auto [output, useAlloca] :
         llvm::zip_equal(outputs, fusionRegionInfo.useAlloca)) {
      if (!useAlloca) {
        initValues.emplace_back(b.tarAddr(output));
      }
    }
    auto tarStride = b.tarValue(kOaccSizeInBytes);

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
        // Extend offset to i64 before multiplication to avoid i32 overflow.
        b.tarJump(
            addr,
            b.tarValue(rewriter.create<arith::MulIOp>(
                loc,
                getElementTypeOrSelf(offset).isInteger(64)
                    ? offset
                    : rewriter
                          .create<arith::ExtSIOp>(loc, rewriter.getI64Type(),
                                                  offset)
                          .getResult(),
                rewriter.create<arith::ConstantIntOp>(
                    loc, rewriter.getI64Type(),
                    mlir::triton::gcu::getBpe(ptrType.getElementType())))));
        map[op] = initValues.size();
        initValues.emplace_back(addr);
      }
    }

    auto loopBody = [&](OpBuilder &builder, Location loc, Value loopIter,
                        ValueRange iterArgs) {
      SmallVector<Value> args(iterArgs);

      // Compute the load/store offset for an alloca-backed buffer at the
      // given inner unroll index. When the outer loop runs more than once
      // (ceil(totalNumElems, vectorLength * loopCnt) > 1), the offset must
      // advance with loopIter so each outer iteration touches a distinct
      // buffer slice. loopIter's unit differs by loop shape: element
      // offset when useLoadStoreInstrOps is non-empty, iteration index
      // otherwise.
      auto computeAllocaOffset = [&](unsigned innerIdx) -> Value {
        Value offset = builder.create<arith::ConstantIndexOp>(
            loc, innerIdx * vectorLength);
        if (ceil<unsigned>(totalNumElems, vectorLength * loopCnt) > 1) {
          Value baseOffset;
          if (!useLoadStoreInstrOps.empty()) {
            // Tar mode: loopIter is an element offset, add it directly.
            baseOffset = loopIter;
          } else {
            // Normal mode: loopIter is an iteration index, multiply by the
            // per-trip block size (vectorLength * loopCnt).
            baseOffset = builder.create<arith::MulIOp>(
                loc, loopIter,
                builder.create<arith::ConstantIndexOp>(loc,
                                                       loopCnt * vectorLength));
          }
          offset = builder.create<arith::AddIOp>(loc, baseOffset, offset);
        }
        return offset;
      };

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
          } else if (isAllocaInput[j]) {
            auto memrefType = cast<MemRefType>(inputs[j].getType());
            auto elementTy = memrefType.getElementType();
            auto totalNumElems = memrefType.getDimSize(0);
            Value offset = computeAllocaOffset(i);
            Value mappingValue;
            if (!elementTy.isInteger(1) && totalNumElems < vectorLength) {
              auto vectorTy =
                  VectorType::get(ArrayRef<int64_t>{totalNumElems}, elementTy);
              Value v = builder
                            .create<vector::LoadOp>(loc, vectorTy, inputs[j],
                                                    ValueRange{offset})
                            .getResult();
              mappingValue =
                  builder
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          elementTy),
                          SmallVector<Value>(vectorLength / totalNumElems, v))
                      ->getResult(0);
            } else {
              auto vectorTy =
                  VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
              mappingValue =
                  builder
                      .create<vector::LoadOp>(loc, vectorTy, inputs[j],
                                              ValueRange{offset})
                      .getResult();
            }
            operandMaps[i].map(op.getRegion().getArgument(j), mappingValue);
          } else if (auto memrefTy =
                         dyn_cast<MemRefType>(inputs[j].getType())) {
            auto elementTy = memrefTy.getElementType();
            Value tarAddr = args[k];
            Value loaded =
                isSmallSize
                    ? loadSmallSizeTarInput(b, builder, elementTy, vectorLength,
                                            tarAddr, tarStride, loc)
                    : generateSplitTarLoad(b, builder, elementTy, vectorLength,
                                           tarAddr, tarStride, loc);
            operandMaps[i].map(op.getRegion().getArgument(j), loaded);
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
          auto constancy = maskedLoadOp->getAttrOfType<IntegerAttr>(kConstancy);
          bool isBroadcast = false;
          bool isFullWarpBroadcast = false;
          if (constancy) {
            auto constancyVal = constancy.getInt();
            auto totalElems = triton::gcu::getTotalElemsPerThread(
                maskedLoadOp.getResult().getType());
            if (constancyVal == static_cast<int>(totalElems)) {
              isBroadcast = true;
              isFullWarpBroadcast = true;
            } else if (constancyVal >= static_cast<int>(vectorLength) &&
                       constancyVal % static_cast<int>(vectorLength) == 0 &&
                       totalNumElems <= vectorLength * loopCnt) {
              isBroadcast = true;
            }
          }
          for (unsigned i = 0; i < loopCnt; ++i) {
            auto result = maskedLoadOp.getResult();
            auto elementTy = result.getType().getElementType();
            auto vecTy =
                VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
            if (isBroadcast) {
              if (isFullWarpBroadcast && i > 0) {
                operandMaps[i].map(result, operandMaps[0].lookup(result));
              } else {
                operandMaps[i].map(
                    result,
                    simplifyLoadToBroadcast(maskedLoadOp, builder,
                                            operandMaps[i], i, vectorLength));
              }
            } else if (useLoadStoreInstrOps.contains(&op) &&
                       (!disableLoadStroreInstrOptimize ||
                        !maskedLoadOp.getMask())) {
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
                        ValueRange{builder.create<arith::IndexCastOp>(
                            loc, builder.getIndexType(),
                            builder.create<arith::AddIOp>(
                                loc,
                                offsets[&op].getType().isInteger(64)
                                    ? offsets[&op]
                                    : builder.create<arith::ExtSIOp>(
                                          loc, builder.getIntegerType(64),
                                          offsets[&op]),
                                builder.create<arith::AddIOp>(
                                    loc,
                                    builder.create<arith::ConstantIntOp>(
                                        loc, builder.getIntegerType(64),
                                        i * vectorLength),
                                    builder.create<arith::IndexCastOp>(
                                        loc, builder.getIntegerType(64),
                                        loopIter))))},
                        mask,
                        builder.create<arith::ConstantOp>(
                            loc, DenseElementsAttr::get(
                                     vecTy, builder.getZeroAttr(elementTy)))));
              }
            } else if (perVecContiguousOps.contains(&op)) {
              operandMaps[i].map(result, simplifyLoadToMaskedLoad(
                                             maskedLoadOp, builder,
                                             operandMaps[i], vectorLength,
                                             needCvtDataLayout, isSmallSize,
                                             totalNumElems, loopCnt, loopIter,
                                             i, !useLoadStoreInstrOps.empty()));
            } else if (offsets.contains(&op)) {
              assert(maskedLoadOp.getMask());
              auto mask = operandMaps[i].lookup(maskedLoadOp.getMask());
              if (getElementTypeOrSelf(mask.getType()).isInteger(8)) {
                mask = builder
                           .create<gcu::VectorConvertOp>(
                               loc,
                               VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               builder.getIntegerType(1)),
                               mask)
                           .getResult(0);
              }
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
                      ValueRange{builder.create<arith::IndexCastOp>(
                          loc, builder.getIndexType(),
                          builder.create<arith::AddIOp>(
                              loc,
                              offsets[&op].getType().isInteger(64)
                                  ? offsets[&op]
                                  : builder.create<arith::ExtSIOp>(
                                        loc, builder.getIntegerType(64),
                                        offsets[&op]),
                              builder.create<arith::AddIOp>(
                                  loc,
                                  builder.create<arith::ConstantIntOp>(
                                      loc, builder.getIntegerType(64),
                                      i * vectorLength),
                                  builder.create<arith::IndexCastOp>(
                                      loc, builder.getIntegerType(64),
                                      loopIter))))},
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
            } else if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(o)) {
              operandMaps[i].map(reshapeOp.getResult(),
                                 operandMaps[i].lookup(reshapeOp.getSrc()));
            } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(o)) {
              operandMaps[i].map(broadcastOp.getResult(),
                                 operandMaps[i].lookup(broadcastOp.getSrc()));
            } else if (auto maskedStoreOp =
                           dyn_cast<triton::gcu::MaskedStoreOp>(o)) {
              if (useLoadStoreInstrOps.contains(&o) &&
                  (!disableLoadStroreInstrOptimize ||
                   !maskedStoreOp.getMask())) {
                auto v = operandMaps[i].lookup(maskedStoreOp.getValue());
                v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);
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
                      ValueRange{builder.create<arith::IndexCastOp>(
                          loc, builder.getIndexType(),
                          builder.create<arith::AddIOp>(
                              loc,
                              offsets[&o].getType().isInteger(64)
                                  ? offsets[&o]
                                  : builder.create<arith::ExtSIOp>(
                                        loc, builder.getIntegerType(64),
                                        offsets[&o]),
                              builder.create<arith::AddIOp>(
                                  loc,
                                  builder.create<arith::ConstantIntOp>(
                                      loc, builder.getIntegerType(64),
                                      i * vectorLength),
                                  builder.create<arith::IndexCastOp>(
                                      loc, builder.getIntegerType(64),
                                      loopIter))))},
                      mask, v);
                }
              } else if (offsets.contains(&o)) {
                assert(maskedStoreOp.getMask());

                auto v = operandMaps[i].lookup(maskedStoreOp.getValue());
                v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);

                auto mask = operandMaps[i].lookup(maskedStoreOp.getMask());
                if (getElementTypeOrSelf(mask.getType()).isInteger(8)) {
                  mask =
                      builder
                          .create<gcu::VectorConvertOp>(
                              loc,
                              VectorType::get(ArrayRef<int64_t>{vectorLength},
                                              builder.getIntegerType(1)),
                              mask)
                          .getResult(0);
                }
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
                    ValueRange{builder.create<arith::IndexCastOp>(
                        loc, builder.getIndexType(),
                        builder.create<arith::AddIOp>(
                            loc,
                            offsets[&o].getType().isInteger(64)
                                ? offsets[&o]
                                : builder.create<arith::ExtSIOp>(
                                      loc, builder.getIntegerType(64),
                                      offsets[&o]),
                            builder.create<arith::AddIOp>(
                                loc,
                                builder.create<arith::ConstantIntOp>(
                                    loc, builder.getIntegerType(64),
                                    i * vectorLength),
                                builder.create<arith::IndexCastOp>(
                                    loc, builder.getIntegerType(64),
                                    loopIter))))},
                    mask, v);
              } else if (perVecContiguousOps.contains(&o)) {
                simplifyStoreToMaskedStore(
                    maskedStoreOp, builder, operandMaps[i], vectorLength,
                    needCvtDataLayout, isSmallSize, totalNumElems, loopCnt,
                    loopIter, i, !useLoadStoreInstrOps.empty());
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
          for (unsigned j = 0, k = 0; j < yieldOp.getNumOperands(); ++j) {
            auto v = operandMaps[i].lookup(yieldOp.getOperand(j));
            if (fusionRegionInfo.useAlloca[j]) {
              Value offset = computeAllocaOffset(i);
              builder.create<vector::StoreOp>(loc, v, outputs[j],
                                              ValueRange{offset});
            } else {
              v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);
              Value tarAddr = args[cnt + k];
              if (isSmallSize)
                v = narrowToStoreVector(builder, loc, v);
              generateSplitTarStore(b, builder, v, tarAddr, tarStride, loc);
              args[cnt + k] = tarAddr;
              ++k;
            }
          }
        }
        builder.create<scf::YieldOp>(loc, args);
      }
    };

    auto setUnrollFullAttr = [&](scf::ForOp forOp) {
      if (!fusionRegionInfo.unrollFull)
        return;
      auto *ctx = rewriter.getContext();
      auto loopUnrollAttr = LLVM::LoopUnrollAttr::get(
          ctx, /*disable=*/{}, /*count=*/{}, /*runtimeDisable=*/{},
          /*full=*/BoolAttr::get(ctx, true), /*followupUnrolled=*/{},
          /*followupRemainder=*/{}, /*followupAll=*/{});
      auto loopAnnotation = LLVM::LoopAnnotationAttr::get(
          ctx, /*disableNonforced=*/{}, /*vectorize=*/{}, /*interleave=*/{},
          /*unroll=*/loopUnrollAttr, /*unrollAndJam=*/{}, /*licm=*/{},
          /*distribute=*/{}, /*pipeline=*/{}, /*peeled=*/{}, /*unswitch=*/{},
          /*mustProgress=*/{}, /*isVectorized=*/{}, /*startLoc=*/{},
          /*endLoc=*/{}, /*parallelAccesses=*/{});
      forOp->setAttr(
          StringAttr::get(ctx, LLVM::LoopAnnotationAttr::getMnemonic()),
          loopAnnotation);
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
              auto forOp = builder.create<scf::ForOp>(
                  loc, lowerBound, upperBound, step, initValues, loopBody);
              setUnrollFullAttr(forOp);
              builder.create<scf::YieldOp>(loc);
            },
            [&](OpBuilder &builder, Location loc) {
              disableLoadStroreInstrOptimize = true;
              auto forOp = builder.create<scf::ForOp>(
                  loc, lowerBound, upperBound, step, initValues, loopBody);
              setUnrollFullAttr(forOp);
              builder.create<scf::YieldOp>(loc);
            });
      } else {
        auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
                                                 step, initValues, loopBody);
        setUnrollFullAttr(forOp);
      }
    } else {
      auto forOp = rewriter.create<scf::ForOp>(
          loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
          rewriter.create<arith::ConstantIndexOp>(
              loc, ceil<unsigned>(totalNumElems, vectorLength * loopCnt)),
          rewriter.create<arith::ConstantIndexOp>(loc, 1), initValues,
          loopBody);
      setUnrollFullAttr(forOp);
    }

    if (auto yieldOp =
            cast<triton::gcu::YieldOp>(op.getRegion().back().getTerminator())) {
      for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
        if (fusionRegionInfo.useAllocaStore[i]) {
          auto result = op.getResult(i);
          auto resultType = dyn_cast<MemRefType>(
              getTypeConverter()->convertType(result.getType()));
          auto lastUser = userAnalysis.getLastUser(result);
          Value out = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                  replaced2Origin, resultType);
          ConfigMatrixStoreLocal(rewriter, loc, resultType, out, results[i]);
          results[i] = out;
        }
      }
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
  Value computeScalarOffsetForVector(Value offset, OpBuilder &builder,
                                     const IRMapping &argMap,
                                     unsigned unrollIdx,
                                     unsigned vectorLength) const {
    IRMapping mapper;
    SetVector<Operation *> visited;
    [[maybe_unused]] bool sawMakeRange = false;
    DenseSet<Operation *> onStack;
    SmallVector<std::pair<Operation *, unsigned>> dfsStack;
    if (auto defOp = offset.getDefiningOp()) {
      dfsStack.push_back({defOp, 0});
      onStack.insert(defOp);
    }
    while (!dfsStack.empty()) {
      Operation *cur = dfsStack.back().first;
      unsigned &operandIdx = dfsStack.back().second;
      if (operandIdx < cur->getNumOperands()) {
        auto operand = cur->getOperand(operandIdx++);
        if (auto child = operand.getDefiningOp()) {
          if (!visited.contains(child) && !onStack.contains(child)) {
            onStack.insert(child);
            dfsStack.push_back({child, 0});
          }
        }
        continue;
      }
      visited.insert(cur);
      onStack.erase(cur);
      dfsStack.pop_back();
    }
    for (auto op : visited) {
      if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
        sawMakeRange = true;
        auto startIdx = makeRangeOp.getStart();
        auto elementTy = makeRangeOp.getResult().getType().getElementType();
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
                      getWarpIds(builder, loc, makeRangeOp.getType()).front()),
                  builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                       totalNumElems)),
              start);
          if (unrollIdx > 0) {
            start = builder.create<arith::AddIOp>(
                loc, start,
                builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                     unrollIdx * vectorLength));
          }
        }
        mapper.map(makeRangeOp.getResult(), start);
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
        mapper.map(splatOp.getResult(), argMap.lookupOrNull(splatOp.getSrc()));
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
        assert(denseAttr && denseAttr.isSplat() &&
               "Constant op should have a splat value");
        mapper.map(constantOp.getResult(),
                   builder.create<arith::ConstantOp>(
                       constantOp.getLoc(), denseAttr.getElementType(),
                       denseAttr.getSplatValue<TypedAttr>()));
      } else {
        auto cloneOp = builder.clone(*op, mapper);
        for (auto [result, newResult] :
             llvm::zip(op->getResults(), cloneOp->getResults())) {
          newResult.setType(getElementTypeOrSelf(result));
          mapper.map(result, newResult);
        }
      }
    }
    assert((unrollIdx == 0 || sawMakeRange) &&
           "per-vector load with unrollIdx > 0 must be MakeRange-indexed");
    return mapper.lookupOrNull(offset);
  }

  Value simplifyLoadToBroadcast(triton::gcu::MaskedLoadOp maskedLoadOp,
                                OpBuilder &builder, const IRMapping &argMap,
                                unsigned vectorIndex,
                                unsigned vectorLength) const {
    auto offset = maskedLoadOp.getOffset();
    auto scalarOffset = computeScalarOffsetForVector(offset, builder, argMap,
                                                     vectorIndex, vectorLength);
    auto loc = maskedLoadOp.getLoc();
    if (!scalarOffset) {
      maskedLoadOp.emitWarning()
          << "simplifyLoadToBroadcast: offset scalarization returned null, "
             "falling back to vector.extract";
      auto vecOffset = argMap.lookup(offset);
      scalarOffset = builder.create<vector::ExtractOp>(loc, vecOffset,
                                                       ArrayRef<int64_t>{0});
    }
    auto v = builder.create<memref::LoadOp>(
        loc, argMap.lookup(maskedLoadOp.getPtr()),
        ValueRange{builder
                       .create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                   scalarOffset)
                       .getResult()});
    return builder.create<vector::BroadcastOp>(
        loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, v.getType()), v);
  }

  // Emit a vector.maskedload for a per-vector-contiguous load.  Within each
  // OACC vector the `vectorLength` elements are contiguous (stride-1), so
  // element 0 of the offset vector gives a valid scalar base address.
  Value simplifyLoadToMaskedLoad(triton::gcu::MaskedLoadOp maskedLoadOp,
                                 OpBuilder &builder, const IRMapping &argMap,
                                 unsigned vectorLength, bool needCvtDataLayout,
                                 bool isSmallSize, unsigned totalNumElems,
                                 unsigned loopCnt, Value loopIter,
                                 unsigned unrollIdx, bool tarMode) const {
    auto offset = maskedLoadOp.getOffset();
    auto loc = maskedLoadOp.getLoc();
    Value scalarOffset = computeScalarOffsetForVector(offset, builder, argMap,
                                                      unrollIdx, vectorLength);
    if (!scalarOffset) {
      maskedLoadOp.emitWarning()
          << "simplifyLoadToMaskedLoad: offset scalarization returned null, "
             "falling back to vector.extract";
      auto vecOffset = argMap.lookup(offset);
      scalarOffset = builder.create<vector::ExtractOp>(loc, vecOffset,
                                                       ArrayRef<int64_t>{0});
    }

    // When the outer loop has more than one iteration, advance the scalar
    // base across outer iterations.  loopIter's unit depends on the region's
    // loop mode: in Tar mode (useLoadStoreInstrOps non-empty) loopIter is an
    // element offset and is added directly; in Normal mode it is an iteration
    // index scaled by the per-trip block size (vectorLength * loopCnt).
    if (totalNumElems > vectorLength * loopCnt) {
      auto offTy = scalarOffset.getType();
      Value iterAdvance;
      if (tarMode) {
        iterAdvance = builder.create<arith::IndexCastOp>(loc, offTy, loopIter);
      } else {
        Value tripStride = builder.create<arith::ConstantIntOp>(
            loc, offTy, static_cast<int64_t>(vectorLength) * loopCnt);
        iterAdvance = builder.create<arith::MulIOp>(
            loc, tripStride,
            builder.create<arith::IndexCastOp>(loc, offTy, loopIter));
      }
      scalarOffset =
          builder.create<arith::AddIOp>(loc, scalarOffset, iterAdvance);
    }

    auto elementTy = maskedLoadOp.getResult().getType().getElementType();
    auto vecTy = VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);

    auto mask = maskedLoadOp.getMask();
    Value maskVal;
    if (!mask) {
      maskVal = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{vectorLength}));
    } else {
      maskVal = argMap.lookup(mask);
      if (needCvtDataLayout) {
        maskVal = builder
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          builder.getIntegerType(1)),
                          maskVal)
                      .getResult(0);
      }
      if (isSmallSize) {
        if (getElementTypeOrSelf(maskVal.getType()).isInteger(8)) {
          maskVal = builder
                        .create<gcu::VectorConvertOp>(
                            loc,
                            VectorType::get(ArrayRef<int64_t>{vectorLength},
                                            builder.getIntegerType(1)),
                            maskVal)
                        .getResult(0);
        }
        maskVal = builder.create<arith::AndIOp>(
            loc, maskVal,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{totalNumElems})));
      }
    }

    auto other = maskedLoadOp.getOther();
    Value passthru = other
                         ? argMap.lookup(other)
                         : builder.create<arith::ConstantOp>(
                               loc, DenseElementsAttr::get(
                                        vecTy, builder.getZeroAttr(elementTy)));
    return builder.create<vector::MaskedLoadOp>(
        loc, vecTy, argMap.lookup(maskedLoadOp.getPtr()),
        ValueRange{builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(), scalarOffset)},
        maskVal, passthru);
  }

  // Emit a vector.maskedstore for a per-vector-contiguous store.  Within each
  // OACC vector the `vectorLength` elements are contiguous (stride-1), so
  // element 0 of the offset vector gives a valid scalar base address.
  void simplifyStoreToMaskedStore(triton::gcu::MaskedStoreOp maskedStoreOp,
                                  OpBuilder &builder, const IRMapping &argMap,
                                  unsigned vectorLength, bool needCvtDataLayout,
                                  bool isSmallSize, unsigned totalNumElems,
                                  unsigned loopCnt, Value loopIter,
                                  unsigned unrollIdx, bool tarMode) const {
    auto offset = maskedStoreOp.getOffset();
    auto loc = maskedStoreOp.getLoc();
    Value scalarOffset = computeScalarOffsetForVector(offset, builder, argMap,
                                                      unrollIdx, vectorLength);
    if (!scalarOffset) {
      maskedStoreOp.emitWarning()
          << "simplifyStoreToMaskedStore: offset scalarization returned null, "
             "falling back to vector.extract";
      auto vecOffset = argMap.lookup(offset);
      scalarOffset = builder.create<vector::ExtractOp>(loc, vecOffset,
                                                       ArrayRef<int64_t>{0});
    }

    if (totalNumElems > vectorLength * loopCnt) {
      auto offTy = scalarOffset.getType();
      Value iterAdvance;
      if (tarMode) {
        iterAdvance = builder.create<arith::IndexCastOp>(loc, offTy, loopIter);
      } else {
        Value tripStride = builder.create<arith::ConstantIntOp>(
            loc, offTy, static_cast<int64_t>(vectorLength) * loopCnt);
        iterAdvance = builder.create<arith::MulIOp>(
            loc, tripStride,
            builder.create<arith::IndexCastOp>(loc, offTy, loopIter));
      }
      scalarOffset =
          builder.create<arith::AddIOp>(loc, scalarOffset, iterAdvance);
    }

    auto mask = maskedStoreOp.getMask();
    Value maskVal;
    if (!mask) {
      maskVal = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{vectorLength}));
    } else {
      maskVal = argMap.lookup(mask);
      if (needCvtDataLayout) {
        maskVal = builder
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          builder.getIntegerType(1)),
                          maskVal)
                      .getResult(0);
      }
      if (isSmallSize) {
        if (getElementTypeOrSelf(maskVal.getType()).isInteger(8)) {
          maskVal = builder
                        .create<gcu::VectorConvertOp>(
                            loc,
                            VectorType::get(ArrayRef<int64_t>{vectorLength},
                                            builder.getIntegerType(1)),
                            maskVal)
                        .getResult(0);
        }
        maskVal = builder.create<arith::AndIOp>(
            loc, maskVal,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{totalNumElems})));
      }
    }

    auto v = argMap.lookup(maskedStoreOp.getValue());
    v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);
    builder.create<vector::MaskedStoreOp>(
        loc, argMap.lookup(maskedStoreOp.getPtr()),
        ValueRange{builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(), scalarOffset)},
        maskVal, v);
  }

  DenseSet<Operation *> trySimplifyLoadStore(
      triton::gcu::ElementwiseFusionRegionOp op, OpBuilder &builder,
      llvm::MapVector<Operation *, Value> &map, Value &mask,
      unsigned vectorLength, DenseSet<Operation *> &perVecContiguousOps) const {
    SetVector<Value> offsets;
    SetVector<Value> masks;
    llvm::MapVector<Value, SmallVector<Operation *>> mask2ops;
    DenseSet<Operation *> useLoadStoreInstrOps;
    mask = nullptr;
    perVecContiguousOps.clear();

    for (auto &o : op.getRegion().back()) {
      if (auto maskedLoadOp = dyn_cast<triton::gcu::MaskedLoadOp>(o)) {
        if (isOffsetContiguousInBlock(&o)) {
          auto offset = maskedLoadOp.getOffset();
          map[&o] = offset;
          offsets.insert(offset);
          if (auto mask = maskedLoadOp.getMask()) {
            masks.insert(mask);
            mask2ops[mask].push_back(&o);
          } else {
            useLoadStoreInstrOps.insert(&o);
          }
        } else if (isOffsetContiguousPerVector(&o, vectorLength)) {
          perVecContiguousOps.insert(&o);
        }
      } else if (auto maskedStoreOp = dyn_cast<triton::gcu::MaskedStoreOp>(o)) {
        if (isOffsetContiguousInBlock(&o)) {
          auto offset = maskedStoreOp.getOffset();
          map[&o] = offset;
          offsets.insert(offset);
          if (auto mask = maskedStoreOp.getMask()) {
            masks.insert(mask);
            mask2ops[mask].push_back(&o);
          } else {
            useLoadStoreInstrOps.insert(&o);
          }
        } else if (isOffsetContiguousPerVector(&o, vectorLength)) {
          perVecContiguousOps.insert(&o);
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
                  predicate == arith::CmpIPredicate::ult ||
                  predicate == arith::CmpIPredicate::sle ||
                  predicate == arith::CmpIPredicate::ule) {
                Value size = builder.create<arith::SubIOp>(loc, rhs, lhs);
                if (predicate == arith::CmpIPredicate::sle ||
                    predicate == arith::CmpIPredicate::ule) {
                  size = builder.create<arith::AddIOp>(
                      loc, size,
                      builder
                          .create<arith::ConstantIntOp>(loc, size.getType(), 1)
                          .getResult());
                }
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
    auto isSupportedCmpOp = [](Operation *op) {
      auto cmpOp = dyn_cast<arith::CmpIOp>(op);
      if (!cmpOp) {
        return false;
      }
      auto predicate = cmpOp.getPredicate();
      return (predicate == arith::CmpIPredicate::slt ||
              predicate == arith::CmpIPredicate::ult ||
              predicate == arith::CmpIPredicate::sge ||
              predicate == arith::CmpIPredicate::uge) &&
             isa_and_nonnull<arith::ConstantOp, triton::SplatOp>(
                 cmpOp.getRhs().getDefiningOp());
    };
    auto isSupportedOp = [](Operation *op) {
      return isa<arith::AndIOp, arith::AddIOp, arith::ExtSIOp, arith::ExtUIOp,
                 arith::ConstantOp, triton::MakeRangeOp, triton::SplatOp>(op);
    };
    for (auto mask : masks) {
      if (auto defOp = mask.getDefiningOp()) {
        if (!isSupportedCmpOp(defOp) && !isSupportedOp(defOp)) {
          mask2ops.erase(mask);
          continue;
        }
        workList.push(defOp);
        opSet.insert(defOp);
        while (!workList.empty()) {
          auto o = workList.front();
          workList.pop();
          for (auto operand : o->getOperands()) {
            auto defOp = operand.getDefiningOp();
            if (defOp) {
              if (isSupportedOp(defOp)) {
                workList.push(defOp);
                opSet.insert(defOp);
              } else if (isSupportedCmpOp(defOp)) {
                workList.push(defOp);
                opSet.insert(defOp);
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

  Value loadSmallSizeTarInput(triton::gcu::TritonGCUBuilder &b,
                              OpBuilder &builder, Type elementTy,
                              unsigned vectorLength, Value &tarAddr,
                              const Value &tarStride, Location loc) const {
    unsigned maxVectorLength =
        4 * kOaccSizeInBytes / mlir::triton::gcu::getBpe(elementTy);
    if (vectorLength <= maxVectorLength) {
      return b.tarLoad(
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy), tarAddr,
          tarStride);
    }

    auto chunkTy =
        VectorType::get(ArrayRef<int64_t>{maxVectorLength}, elementTy);
    Value chunk = b.tarLoad(chunkTy, tarAddr, tarStride);
    unsigned numParts = vectorLength / maxVectorLength;
    SmallVector<Value> parts;
    parts.push_back(chunk);
    auto zeroChunk = builder.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(chunkTy, builder.getZeroAttr(elementTy)));
    for (unsigned p = 1; p < numParts; ++p)
      parts.push_back(zeroChunk);
    SmallVector<Type> resultTypes;
    resultTypes.push_back(
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy));
    return builder.create<gcu::VectorConvertOp>(loc, resultTypes, parts)
        .getResult(0);
  }

  Value generateSplitTarLoad(triton::gcu::TritonGCUBuilder &b,
                             OpBuilder &builder, Type elementTy,
                             unsigned vectorLength, Value &tarAddr,
                             const Value &tarStride, Location loc) const {
    unsigned bpe = mlir::triton::gcu::getBpe(elementTy);
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / bpe;

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
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / bpe;

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
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

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
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

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
      triton_gcu::compat::createVectorScatterOp(
          builder, loc, map.lookup(op.getPtr()), ValueRange{zero},
          offsets.getResult(i), mask, values.getResult(i));
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
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

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
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

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
    triton_gcu::compat::createVectorScatterOp(
        builder, loc, map.lookup(op.getPtr()), ValueRange{zero},
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
    Value result;
    auto resultTy = VectorType::get(ArrayRef<int64_t>{vectorLength},
                                    getElementTypeOrSelf(op.getResult()));
    if (mlir::triton::gcu::isNvLibDeviceSymbol(symbol)) {
      std::string efSymbol = "__ef_v";
      efSymbol += symbol.drop_front(strlen("__nv_"));
      result = builder.create<gcu::ExternElementwiseOp>(loc, resultTy, operands,
                                                        efSymbol);
    } else if (mlir::triton::gcu::isMixedPrecisionSymbol(symbol)) {
      result = builder.create<gcu::ExternElementwiseOp>(loc, resultTy, operands,
                                                        symbol);
    } else {
      llvm_unreachable(
          ("unsupported extern elementwise: " + symbol).str().c_str());
    }
    map.map(op.getResult(), result);
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
        maxVectorLength = 4 * kOaccSizeInBytes / bpe;
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
        auto inValue = map.lookup(cvtOp.getIn());
        if (getElementTypeOrSelf(inValue.getType()).isInteger(1)) {
          inValue = builder
                        .create<gcu::VectorConvertOp>(
                            op.getLoc(),
                            VectorType::get(ArrayRef<int64_t>{vectorLength},
                                            builder.getIntegerType(8)),
                            inValue)
                        .getResult(0);
        }
        map.map(cvtOp.getOut(), inValue);
        return;
      } else {
        auto outElementType = getElementTypeOrSelf(cvtOp.getOut().getType());
        unsigned outBpe = mlir::triton::gcu::getBpe(outElementType);
        unsigned maxVectorLength = 4 * kOaccSizeInBytes / outBpe;
        if (vectorLength > maxVectorLength &&
            getElementTypeOrSelf(cvtOp.getIn().getType()).isInteger(1)) {
          newOp = generateSplitExtUIOp(cvtOp, builder, map, vectorLength,
                                       maxVectorLength);
        } else {
          newOp = builder.clone(op, map);
        }
      }
    } else if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      auto lhs = op.getOperand(0);
      auto rhs = op.getOperand(1);
      auto lhsValue = map.lookup(lhs);
      auto rhsValue = map.lookup(rhs);
      if (getElementTypeOrSelf(lhsValue.getType()).isInteger(1) &&
          getElementTypeOrSelf(rhsValue.getType()).isInteger(8)) {
        map.map(lhs, builder
                         .create<gcu::VectorConvertOp>(
                             op.getLoc(),
                             VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             builder.getIntegerType(8)),
                             lhsValue)
                         .getResult(0));
        newOp = builder.clone(op, map);
        map.map(lhs, lhsValue);
      } else if (getElementTypeOrSelf(lhsValue.getType()).isInteger(8) &&
                 getElementTypeOrSelf(rhsValue.getType()).isInteger(1)) {
        map.map(rhs, builder
                         .create<gcu::VectorConvertOp>(
                             op.getLoc(),
                             VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             builder.getIntegerType(8)),
                             rhsValue)
                         .getResult(0));
        newOp = builder.clone(op, map);
        map.map(rhs, rhsValue);
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
      if (isa<arith::CmpFOp, arith::CmpIOp>(op) && needCvtDataLayout &&
          llvm::any_of(op.getUsers(), [&](Operation *user) {
            if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
              return selectOp.getCondition() != op.getResult(0);
            } else if (auto maskedLoadOp =
                           dyn_cast<triton::gcu::MaskedLoadOp>(user)) {
              return maskedLoadOp.getMask() != op.getResult(0);
            } else if (auto maskedStoreOp =
                           dyn_cast<triton::gcu::MaskedStoreOp>(user)) {
              return maskedStoreOp.getMask() != op.getResult(0);
            }
            return !isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(user);
          })) {
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
