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

#ifndef KURAMA_TRITONGCUASYNC_TO_GCU_H_
#define KURAMA_TRITONGCUASYNC_TO_GCU_H_

#include "Conversion/TritonToGCU/Utility.h"
#include "TritionToGCUBase.h"

#include <map>

#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/MathExt/IR/MathExt.h"
#include "Dialect/MathExt/IR/MathExtTypes.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

void getPipelineAsyncResourceMaping(
    Operation *module, std::map<Operation *, Operation *> &asyncLoad2Tag,
    llvm::DenseMap<Operation *, Value> &asyncLoad2Tagidex,
    std::map<Operation *, Operation *> &asyncWait2Tag) {
  int32_t pipelineResourceNumber = -1;
  std::map<Operation *, Operation *> shareAlloc2Tags;
  module->walk<WalkOrder::PreOrder>([&](triton::gcu::BufferAllocOp op) {
    auto type = dyn_cast<triton::gpu::MemDescType>(op.getType());
    int32_t dim0Size = type.getShape()[0];
    if ((pipelineResourceNumber != -1) &&
        (pipelineResourceNumber != dim0Size)) {
      assert(false && " all triton::gcu::BufferAllocOp  should has some "
                      "PipelineResourceNumber!!!");
    }
    pipelineResourceNumber = dim0Size;
    OpBuilder builder(op.getOperation());
    auto tagType = MemRefType::get(ArrayRef<int64_t>{pipelineResourceNumber},
                                   builder.getI32Type());
    auto sharedTags = builder.create<memref::AllocOp>(op.getLoc(), tagType);
    sharedTags->setAttr("gcu.share_tag", builder.getUnitAttr());
    shareAlloc2Tags[op.getOperation()] = sharedTags.getOperation();
  });

  auto getShareAlloc = [&](Operation *tokenDefineOp) {
    assert(isa<triton::gcu::AsyncLoadFromGlobalOp>(tokenDefineOp) &&
           " wait_op's tag should be a AsyncLoadFromGlobalOp op!");
    auto dstbuffer =
        dyn_cast<triton::gcu::AsyncLoadFromGlobalOp>(tokenDefineOp).getDstMem();
    auto bufferDefineOp = dstbuffer.getDefiningOp();
    if (!bufferDefineOp ||
        !isa<triton::gpu::MemDescSubsliceOp>(bufferDefineOp)) {
      assert(false && " AsyncLoadFromGlobalOp's dst should be a subview op!");
    }
    auto subView = dyn_cast<triton::gpu::MemDescSubsliceOp>(bufferDefineOp);
    auto shareAllocOp = subView.getSrc().getDefiningOp();
    if (!shareAllocOp || !isa<triton::gcu::BufferAllocOp>(shareAllocOp)) {
      assert(false && " MemDescSubsliceOp's src should be a BufferAllocOp op!");
    }
    return shareAllocOp;
  };

  module->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    llvm::TypeSwitch<mlir::Operation *>(operation)
        .Case<triton::gcu::AsyncLoadFromGlobalOp>(
            [&](triton::gcu::AsyncLoadFromGlobalOp load) {
              auto dstbuffer = load.getDstMem();
              auto defineOp = dstbuffer.getDefiningOp();
              if (!defineOp || !isa<triton::gpu::MemDescSubsliceOp>(defineOp)) {
                assert(false &&
                       " AsyncLoadFromGlobalOp's dst should be a subview op!");
              }
              auto subView = dyn_cast<triton::gpu::MemDescSubsliceOp>(defineOp);
              auto shareAllocOp = subView.getSrc().getDefiningOp();
              if (!shareAllocOp ||
                  !isa<triton::gcu::BufferAllocOp>(shareAllocOp)) {
                assert(false && " MemDescSubsliceOp's src should be a "
                                "BufferAllocOp op!");
              }
              asyncLoad2Tag[operation] = shareAlloc2Tags[shareAllocOp];
              auto opOffsets = subView.getOffsets();
              OpBuilder builder(subView);
              SmallVector<Value> opOffsetVals;
              for (int offset : opOffsets) {
                opOffsetVals.push_back(builder.create<arith::ConstantIntOp>(
                    subView.getLoc(), offset, 32));
              }
              asyncLoad2Tagidex[operation] = opOffsetVals[0];
            })
        .Case<triton::gcu::AsyncWaitOp>([&](triton::gcu::AsyncWaitOp wait) {
          auto waitToken = wait.getAsyncToken()[0];
          if (auto tocken = dyn_cast<BlockArgument>(waitToken)) {
            auto waitParent = operation->getParentOp();
            if (isa<scf::IfOp>(waitParent)) {
              waitParent = waitParent->getParentOp();
            }
            assert(isa<scf::ForOp>(waitParent) &&
                   "if async wait got a block argument, it should be in ForOp");
            auto forInitToken =
                dyn_cast<scf::ForOp>(waitParent).getTiedLoopInit(tocken)->get();
            auto tokenDefineOp = forInitToken.getDefiningOp();
            if (tokenDefineOp) {
              asyncWait2Tag[operation] =
                  shareAlloc2Tags[getShareAlloc(tokenDefineOp)];
            }
          } else {
            auto tokenDefineOp = waitToken.getDefiningOp();
            if (tokenDefineOp) {
              asyncWait2Tag[operation] =
                  shareAlloc2Tags[getShareAlloc(tokenDefineOp)];
            }
          }
        });
  });
}

struct TTBufferAllocOpLowering
    : SharedConversionPattern<triton::gcu::BufferAllocOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::BufferAllocOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, alloc.getOperation());
    if (pTagPool.isExistInMap(alloc.getOperation())) {
      pTagPool.releaseMap(alloc.getOperation());
    }
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(alloc.getType()));
    auto output = rewriter.create<memref::AllocOp>(alloc.getLoc(), resultType);
    leaveTritionOp(rewriter, alloc.getOperation());
    rewriter.replaceOp(alloc, output);
    return success();
  }
};

struct TTBufferDeallocOpLowering
    : SharedConversionPattern<triton::gcu::BufferDeallocOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::BufferDeallocOp dealloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, dealloc.getOperation());
    if (pTagPool.isExistInMap(dealloc.getOperation())) {
      pTagPool.releaseMap(dealloc.getOperation());
    }
    rewriter.create<memref::DeallocOp>(dealloc.getLoc(), adaptor.getSrc());
    leaveTritionOp(rewriter, dealloc.getOperation());
    rewriter.eraseOp(dealloc);
    return success();
  }
};

struct TTLocalLoadOpLowering
    : SharedConversionPattern<triton::gcu::LocalLoadOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::LocalLoadOp toTensor, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, toTensor.getOperation());
    if (pTagPool.isExistInMap(toTensor.getOperation())) {
      pTagPool.releaseMap(toTensor.getOperation());
    }
    rewriter.replaceOp(toTensor, adaptor.getSrc());
    return success();
  }
};

inline Value dot(RewriterBase &rewriter, Location loc, ArrayRef<Value> offsets,
                 ArrayRef<Value> strides) {
  assert(offsets.size() == strides.size());
  Value ret = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    ret = rewriter.create<arith::AddIOp>(
        loc, ret, rewriter.create<arith::MulIOp>(loc, offset, stride));
  }
  return ret;
}

struct TTMemDescSubsliceOpLowering
    : SharedConversionPattern<triton::gpu::MemDescSubsliceOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescSubsliceOp subview, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, subview.getOperation());
    if (pTagPool.isExistInMap(subview.getOperation())) {
      pTagPool.releaseMap(subview.getOperation());
    }
    auto resultType = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(subview.getType()));
    auto loc = subview.getLoc();
    auto src = adaptor.getSrc();
    auto sourceType = dyn_cast<MemRefType>(src.getType());
    auto sourceRank = sourceType.getRank();
    auto elemType = resultType.getElementType();
    auto [strides, offset] = sourceType.getStridesAndOffset();
    (void)offset;
    auto opOffsets = subview.getOffsets();
    SmallVector<Value> opOffsetVals;
    for (long int offset : opOffsets) {
      opOffsetVals.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, offset, 32));
    }
    assert((opOffsetVals.size() == strides.size()) &&
           "offset size is not equal to stride size !!!");
    assert((opOffsetVals.size() == static_cast<unsigned>(sourceRank)) &&
           "offset size is not equal to rank !!!");
    // SmallVector<OpFoldResult> outOffsets;
    SmallVector<OpFoldResult> strideVals;
    SmallVector<Value> strideValues;
    for (int32_t i = 0; i < sourceRank; i++) {
      if (i > 0) {
        strideVals.push_back(rewriter.getIndexAttr(strides[i]));
      }
      strideValues.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, strides[i], 32));
    }

    auto finalOffsetValue = dot(rewriter, loc, opOffsetVals, strideValues);
    auto bpe = elemType.getIntOrFloatBitWidth() / 8;
    auto elementType = resultType.getElementType();
    int64_t size = 1;
    for (int i = 0; i < sourceType.getRank(); i++) {
      size *= sourceType.getShape()[i];
    }
    // Create flattened buffer
    MemRefType flatType = MemRefType::get({size}, elementType, AffineMap{},
                                          resultType.getMemorySpace());
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value flatBuffer = rewriter.create<memref::ReinterpretCastOp>(
        loc, flatType, src, zero,
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, size)},
        ValueRange{one});
    auto ptrType = gcu::PtrType::get(getContext(), elementType);
    Value ptr = rewriter.create<gcu::MemRefToPtrOp>(loc, ptrType, flatBuffer);
    MemRefType memType1D =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());
    auto buffer1D = rewriter.create<gcu::PtrToMemRefOp>(loc, memType1D, ptr);

    auto I8Offset = rewriter.create<arith::MulIOp>(
        loc, finalOffsetValue,
        rewriter.create<arith::ConstantIntOp>(loc, bpe, 32));
    auto bufferWithSpace = rewriter.create<memref::MemorySpaceCastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        AffineMap{}, resultType.getMemorySpace()),
        buffer1D);
    auto output = rewriter.create<memref::ViewOp>(
        loc, resultType, bufferWithSpace,
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                            I8Offset),
        ValueRange{});
    leaveTritionOp(rewriter, subview.getOperation());
    rewriter.replaceOp(subview, output);
    return success();
  }
};

struct TTAsyncLoadFromGlobalOpLowering
    : SharedConversionPattern<triton::gcu::AsyncLoadFromGlobalOp> {
  using SharedConversionPattern::SharedConversionPattern;

  std::map<Operation *, Operation *> &asyncLoad2Tag;
  llvm::DenseMap<Operation *, Value> &asyncLoad2Tagidex;
  explicit TTAsyncLoadFromGlobalOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      triton::gcu::FirstLastUserAnalysis &userAnalysis,
      std::map<Operation *, Operation *> &replaced2Origin,
      triton::gcu::PrivateTagPool &pTagPool,
      std::map<Operation *, Operation *> &inAsyncLoad2Tags,
      llvm::DenseMap<Operation *, Value> &inAsyncLoad2Tagidex)
      : SharedConversionPattern<triton::gcu::AsyncLoadFromGlobalOp>(
            converter, ctx, userAnalysis, replaced2Origin, pTagPool),
        asyncLoad2Tag(inAsyncLoad2Tags),
        asyncLoad2Tagidex(inAsyncLoad2Tagidex) {}

  LogicalResult
  matchAndRewrite(triton::gcu::AsyncLoadFromGlobalOp asyncLoad,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, asyncLoad.getOperation());
    if (pTagPool.isExistInMap(asyncLoad.getOperation())) {
      pTagPool.releaseMap(asyncLoad.getOperation());
    }
    bool isPrologueLoad = false;
    if (asyncLoad.getOperation()->getAttr("Prologue_stage_idex")) {
      isPrologueLoad = true;
    }
    auto loc = asyncLoad.getLoc();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto outputBuffer = adaptor.getDstMem();
    auto outputType = dyn_cast<MemRefType>(outputBuffer.getType());
    auto elemType = outputType.getElementType();
    auto rank = outputType.getRank();
    SmallVector<Value, 4> sourceShape;
    sourceShape.push_back(adaptor.getShape()[0]);
    for (unsigned i = 0; i < rank - 1; ++i) {
      sourceShape.push_back(rewriter.create<arith::DivSIOp>(
          loc, adaptor.getStrides()[i], adaptor.getStrides()[i + 1]));
    }
    SmallVector<Value, 4> offsets;
    for (unsigned i = 0; i < adaptor.getOffsets().size(); ++i) {
      auto offset = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), adaptor.getOffsets()[i]);
      offsets.push_back(offset);
    }
    SmallVector<Value, 4> sliceShape;
    for (unsigned i = 0; i < rank; ++i) {
      sliceShape.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), adaptor.getShape()[i]));
    }
    assert(
        (asyncLoad2Tag.find(asyncLoad.getOperation()) != asyncLoad2Tag.end()) &&
        "AsyncLoadFromGlobalOp had no mapping tags !!!");
    assert((asyncLoad2Tagidex.find(asyncLoad.getOperation()) !=
            asyncLoad2Tagidex.end()) &&
           "AsyncLoadFromGlobalOp had no mapping tagindx !!!");
    if (isPrologueLoad == true) {
      int32_t prologueIdx =
          dyn_cast<IntegerAttr>(
              asyncLoad.getOperation()->getAttr("Prologue_stage_idex"))
              .getInt();
      // get range from for
      Operation *forUser = nullptr;
      int32_t userNumber = 0;
      for (Operation *user : asyncLoad.getOperation()->getUsers()) {
        userNumber++;
        if (isa<scf::ForOp>(user)) {
          forUser = user;
        }
      }
      if (forUser == nullptr || userNumber > 2) {
        asyncLoad.dump();
        assert(false && "please carefully check pingpong prologue flow!!!!");
      }
      auto forOp = llvm::dyn_cast<scf::ForOp>(forUser);
      auto step = forOp.getStep();
      auto upperBound = forOp.getUpperBound();
      auto lowerBound = forOp.getLowerBound();

      auto forRange =
          rewriter.create<arith::SubIOp>(loc, upperBound, lowerBound);
      auto reminAdd = rewriter.create<arith::SubIOp>(
          loc, step,
          rewriter.create<arith::ConstantOp>(
              step.getLoc(), rewriter.getIntegerAttr(step.getType(), 1)));
      auto forStepNum = rewriter.create<arith::DivSIOp>(
          loc,
          rewriter.create<arith::AddIOp>(
              loc, rewriter.create<math::AbsIOp>(loc, forRange), reminAdd),
          step);
      auto isSmallThan = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt,
          rewriter.create<arith::ConstantOp>(
              step.getLoc(),
              rewriter.getIntegerAttr(step.getType(), prologueIdx)),
          forStepNum);
      rewriter.create<scf::IfOp>(
          loc, isSmallThan, [&](OpBuilder &builder, Location loc) {
            auto isThread0 = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq,
                builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
            auto defaultValue =
                asyncLoad.getDefaultValue()
                    ? adaptor.getDefaultValue()
                    : triton::gcu::createConstantZero(builder, loc, elemType);
            auto tagIdx = builder
                              .create<arith::IndexCastOp>(
                                  loc, builder.getIndexType(),
                                  asyncLoad2Tagidex[asyncLoad.getOperation()])
                              .getResult();
            builder.create<scf::IfOp>(
                loc, isThread0, [&](OpBuilder &builder, Location loc) {
                  ConfigGcuLoad(
                      builder, loc, pTagPool, adaptor.getDstMem(),
                      asyncLoad.getOperation(), outputType, adaptor.getPtr(),
                      adaptor.getStrides(), adaptor.getShape(), defaultValue,
                      triton::gcu::TagInfo(
                          asyncLoad2Tag[asyncLoad.getOperation()]->getResult(0),
                          tagIdx, true),
                      true);
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
      leaveTritionOp(rewriter, asyncLoad.getOperation());
      rewriter.replaceOp(asyncLoad,
                         asyncLoad2Tagidex[asyncLoad.getOperation()]);
      return success();
    }
    // to avoid share momeory race
    rewriter.create<gpu::BarrierOp>(loc);
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
    auto defaultValue =
        asyncLoad.getDefaultValue()
            ? adaptor.getDefaultValue()
            : triton::gcu::createConstantZero(rewriter, loc, elemType);
    auto tagIdx = rewriter
                      .create<arith::IndexCastOp>(
                          loc, rewriter.getIndexType(),
                          asyncLoad2Tagidex[asyncLoad.getOperation()])
                      .getResult();
    rewriter.create<scf::IfOp>(
        loc, isThread0, [&](OpBuilder &builder, Location loc) {
          ConfigGcuLoad(
              builder, loc, pTagPool, adaptor.getDstMem(),
              asyncLoad.getOperation(), outputType, adaptor.getPtr(),
              adaptor.getStrides(), adaptor.getShape(), defaultValue,
              triton::gcu::TagInfo(
                  asyncLoad2Tag[asyncLoad.getOperation()]->getResult(0), tagIdx,
                  true),
              true);
          builder.create<scf::YieldOp>(loc);
        });
    leaveTritionOp(rewriter, asyncLoad.getOperation());
    rewriter.replaceOp(asyncLoad, asyncLoad2Tagidex[asyncLoad.getOperation()]);
    return success();
  }
};

struct TTAsyncWaitOpLowering
    : SharedConversionPattern<triton::gcu::AsyncWaitOp> {
  using SharedConversionPattern::SharedConversionPattern;
  std::map<Operation *, Operation *> &asyncWait2Tag;

  explicit TTAsyncWaitOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      triton::gcu::FirstLastUserAnalysis &userAnalysis,
      std::map<Operation *, Operation *> &replaced2Origin,
      triton::gcu::PrivateTagPool &pTagPool,
      std::map<Operation *, Operation *> &inAsyncWait2Tag)
      : SharedConversionPattern<triton::gcu::AsyncWaitOp>(
            converter, ctx, userAnalysis, replaced2Origin, pTagPool),
        asyncWait2Tag(inAsyncWait2Tag) {}

  LogicalResult
  matchAndRewrite(triton::gcu::AsyncWaitOp wait, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, wait.getOperation());
    if (pTagPool.isExistInMap(wait.getOperation())) {
      pTagPool.releaseMap(wait.getOperation());
    }
    auto loc = wait.getLoc();
    assert((asyncWait2Tag.find(wait.getOperation()) != asyncWait2Tag.end()) &&
           "AsyncWaitOp had no mapping tags !!!");
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
    rewriter.create<scf::IfOp>(
        loc, isThread0, [&](OpBuilder &builder, Location loc) {
          auto tagIdx =
              rewriter
                  .create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                              adaptor.getAsyncToken()[0])
                  .getResult();
          WaitGcuLoadStore(builder, loc,
                           triton::gcu::TagInfo(
                               asyncWait2Tag[wait.getOperation()]->getResult(0),
                               tagIdx, true),
                           zero);
          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<gpu::BarrierOp>(loc);
    leaveTritionOp(rewriter, wait.getOperation());
    rewriter.replaceOp(wait, adaptor.getAsyncToken()[0]);
    return success();
  }
};
#endif
