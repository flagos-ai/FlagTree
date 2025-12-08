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

#include "TritionToGCUBase.h"
#include "TritonGCUToGCUUtils.h"

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
  module->walk<WalkOrder::PreOrder>([&](triton::gcu::ShareAllocOp op) {
    auto type = dyn_cast<triton::gpu::MemDescType>(op.getType());
    int32_t dim0Size = type.getShape()[0];
    if ((pipelineResourceNumber != -1) &&
        (pipelineResourceNumber != dim0Size)) {
      assert(false && " all triton::gcu::ShareAllocOp  should has some "
                      "PipelineResourceNumber!!!");
    }
    pipelineResourceNumber = dim0Size;
    OpBuilder builder(op.getOperation());
    auto tagType = MemRefType::get(ArrayRef<int64_t>{pipelineResourceNumber},
                                   builder.getI32Type());
    auto tag = builder.create<memref::AllocOp>(op.getLoc(), tagType);
    tag->setAttr("gcu.share_tag", builder.getUnitAttr());
    shareAlloc2Tags[op.getOperation()] = tag.getOperation();
  });

  auto getShareAlloc = [&](Operation *tokenDefineOp) {
    assert(isa<triton::gcu::AsyncLoadGlobalToShareOp>(tokenDefineOp) &&
           " wait_op's tag should be a AsyncLoadGlobalToShareOp op!");
    auto dstbuffer =
        dyn_cast<triton::gcu::AsyncLoadGlobalToShareOp>(tokenDefineOp)
            .getDstMem();
    auto bufferDefineOp = dstbuffer.getDefiningOp();
    if (!bufferDefineOp ||
        !isa<triton::gpu::MemDescSubviewOp>(bufferDefineOp)) {
      assert(false &&
             " AsyncLoadGlobalToShareOp's dst should be a subview op!");
    }
    auto subView = dyn_cast<triton::gpu::MemDescSubviewOp>(bufferDefineOp);
    auto shareAllocOp = subView.getSrc().getDefiningOp();
    if (!shareAllocOp || !isa<triton::gcu::ShareAllocOp>(shareAllocOp)) {
      assert(false && " MemDescSubviewOp's src should be a ShareAllocOp op!");
    }
    return shareAllocOp;
  };

  module->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    llvm::TypeSwitch<mlir::Operation *>(operation)
        .Case<triton::gcu::AsyncLoadGlobalToShareOp>(
            [&](triton::gcu::AsyncLoadGlobalToShareOp load) {
              auto dstbuffer = load.getDstMem();
              auto defineOp = dstbuffer.getDefiningOp();
              if (!defineOp || !isa<triton::gpu::MemDescSubviewOp>(defineOp)) {
                assert(
                    false &&
                    " AsyncLoadGlobalToShareOp's dst should be a subview op!");
              }
              auto subView = dyn_cast<triton::gpu::MemDescSubviewOp>(defineOp);
              auto shareAllocOp = subView.getSrc().getDefiningOp();
              if (!shareAllocOp ||
                  !isa<triton::gcu::ShareAllocOp>(shareAllocOp)) {
                assert(false &&
                       " MemDescSubviewOp's src should be a ShareAllocOp op!");
              }
              asyncLoad2Tag[operation] = shareAlloc2Tags[shareAllocOp];
              SmallVector<Value> opOffsetVals = subView.getOffsets();
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

struct TTShareAllocOpLowering : OpConversionPattern<triton::gcu::ShareAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::ShareAllocOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, alloc.getOperation());
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(alloc.getType()));
    auto output = rewriter.create<memref::AllocOp>(alloc.getLoc(), resultType);
    leaveTritionOp(rewriter, alloc.getOperation());
    rewriter.replaceOp(alloc, output);
    return success();
  }
};

struct TTShareDeallocOpLowering
    : OpConversionPattern<triton::gcu::ShareDeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::ShareDeallocOp dealloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, dealloc.getOperation());
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
  matchAndRewrite(triton::gcu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcLayout =
        cast<triton::gpu::TensorOrMemDesc>(op.getSrc().getType()).getEncoding();
    auto dstLayout = dyn_cast<RankedTensorType>(op.getType()).getEncoding();
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    auto firstUser = userAnalysis.getFirstUserOp(op.getOperation());
    auto tag = (firstUser == nullptr) ? getPrivateDTETag(rewriter, op)
                                      : createPrivateDTETag(rewriter, op);
    // share to Distributed
    if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
      // copy to local
      auto output = loadFromSharedMem(rewriter, tag, op.getResult().getType(),
                                      adaptor.getSrc(), false, lastUser,
                                      firstUser, userAnalysis, replaced2Origin);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
               isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      // Distributed to dot operand
      // to dot a or b
      auto output = loadFromSharedMemForDotOperand(
          rewriter, tag, op.getResult().getType(),
          op.getSrc().getType().getShape(), adaptor.getSrc(), lastUser,
          firstUser, userAnalysis, replaced2Origin);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else {
      op.dump();
      llvm::report_fatal_error(
          "[Error] gcu::LocalLoadOp maybe had bad used in pinpong\n");
    }
    return success();
  }
};

inline Value dot(RewriterBase &rewriter, Location loc, ArrayRef<Value> offsets,
                 ArrayRef<Value> strides) {
  assert(offsets.size() == strides.size());
  Value ret =
      rewriter.create<arith::ConstantIntOp>(loc, 0, rewriter.getI32Type());
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    ret = rewriter.create<arith::AddIOp>(
        loc, ret, rewriter.create<arith::MulIOp>(loc, offset, stride));
  }
  return ret;
}

struct TTMemDescSubviewOpLowering
    : OpConversionPattern<triton::gpu::MemDescSubviewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescSubviewOp subview, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, subview.getOperation());
    auto resultType = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(subview.getType()));
    auto loc = subview.getLoc();
    auto src = adaptor.getSrc();
    auto sourceType = dyn_cast<MemRefType>(src.getType());
    auto sourceRank = sourceType.getRank();
    auto [strides, offset] = sourceType.getStridesAndOffset();
    (void)offset;
    SmallVector<Value> opOffsetVals = subview.getOffsets();
    assert((opOffsetVals.size() == strides.size()) &&
           "offset size is not equal to stride size !!!");
    assert((opOffsetVals.size() == static_cast<unsigned>(sourceRank)) &&
           "offset size is not equal to rank !!!");

    auto elemType = resultType.getElementType();
    // SmallVector<OpFoldResult> outOffsets;
    SmallVector<OpFoldResult> strideVals;
    SmallVector<Value> strideValues;
    for (int32_t i = 0; i < sourceRank; i++) {
      if (i > 0) {
        strideVals.push_back(rewriter.getIndexAttr(strides[i]));
      }
      strideValues.push_back(rewriter.create<arith::ConstantIntOp>(
          loc, strides[i], opOffsetVals[0].getType()));
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
        rewriter.create<arith::ConstantIntOp>(loc, bpe,
                                              opOffsetVals[0].getType()));
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

struct TTAsyncLoadGlobalToShareOpLowering
    : OpConversionPattern<triton::gcu::AsyncLoadGlobalToShareOp> {
  using OpConversionPattern::OpConversionPattern;

  std::map<Operation *, Operation *> &asyncLoad2Tag;
  llvm::DenseMap<Operation *, Value> &asyncLoad2Tagidex;
  explicit TTAsyncLoadGlobalToShareOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      std::map<Operation *, Operation *> &inAsyncLoad2Tags,
      llvm::DenseMap<Operation *, Value> &inAsyncLoad2Tagidex)
      : OpConversionPattern<triton::gcu::AsyncLoadGlobalToShareOp>(converter,
                                                                   ctx),
        asyncLoad2Tag(inAsyncLoad2Tags),
        asyncLoad2Tagidex(inAsyncLoad2Tagidex) {}

  LogicalResult
  matchAndRewrite(triton::gcu::AsyncLoadGlobalToShareOp asyncLoad,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, asyncLoad.getOperation());
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
        "AsyncLoadGlobalToShareOp had no mapping tags !!!");
    assert((asyncLoad2Tagidex.find(asyncLoad.getOperation()) !=
            asyncLoad2Tagidex.end()) &&
           "AsyncLoadGlobalToShareOp had no mapping tagindx !!!");
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
            auto outTransType = MemRefType::get(outputType.getShape(),
                                                outputType.getElementType());
            auto outTrans = builder.create<memref::AllocOp>(loc, outTransType);
            rewriter.create<scf::IfOp>(
                loc, isThread0, [&](OpBuilder &builder, Location loc) {
                  ConfigGcuLoad(
                      rewriter, loc, adaptor.getDstMem(), outTrans,
                      asyncLoad.getOperation(), outputType, adaptor.getPtr(),
                      adaptor.getStrides(), adaptor.getShape(), defaultValue,
                      asyncLoad2Tag[asyncLoad.getOperation()]->getResult(0),
                      tagIdx, true);
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<memref::DeallocOp>(loc, outTrans);
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
    auto outTransType =
        MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outTrans = rewriter.create<memref::AllocOp>(loc, outTransType);
    rewriter.create<scf::IfOp>(
        loc, isThread0, [&](OpBuilder &builder, Location loc) {
          ConfigGcuLoad(rewriter, loc, adaptor.getDstMem(), outTrans,
                        asyncLoad.getOperation(), outputType, adaptor.getPtr(),
                        adaptor.getStrides(), adaptor.getShape(), defaultValue,
                        asyncLoad2Tag[asyncLoad.getOperation()]->getResult(0),
                        tagIdx, true);
          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<memref::DeallocOp>(loc, outTrans);
    leaveTritionOp(rewriter, asyncLoad.getOperation());
    rewriter.replaceOp(asyncLoad, asyncLoad2Tagidex[asyncLoad.getOperation()]);
    return success();
  }
};

struct TTAsyncWaitOpLowering : OpConversionPattern<triton::gcu::AsyncWaitOp> {
  using OpConversionPattern::OpConversionPattern;
  std::map<Operation *, Operation *> &asyncWait2Tag;

  explicit TTAsyncWaitOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      std::map<Operation *, Operation *> &inAsyncWait2Tag)
      : OpConversionPattern<triton::gcu::AsyncWaitOp>(converter, ctx),
        asyncWait2Tag(inAsyncWait2Tag) {}

  LogicalResult
  matchAndRewrite(triton::gcu::AsyncWaitOp wait, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, wait.getOperation());
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
                           asyncWait2Tag[wait.getOperation()]->getResult(0),
                           tagIdx, zero);
          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<gpu::BarrierOp>(loc);
    leaveTritionOp(rewriter, wait.getOperation());
    rewriter.replaceOp(wait, adaptor.getAsyncToken()[0]);
    return success();
  }
};
#endif
