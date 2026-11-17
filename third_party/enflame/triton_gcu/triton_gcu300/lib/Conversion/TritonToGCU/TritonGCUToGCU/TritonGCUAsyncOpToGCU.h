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
#include <utility>

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

Value getTransBuffer(RewriterBase &rewriter, Location loc, Value TransBuffers,
                     Value tagindex) {
  auto buffertype = cast<MemRefType>(TransBuffers.getType());
  auto shapes = buffertype.getShape();
  auto rank = buffertype.getRank();
  auto elementType = buffertype.getElementType();
  auto bpe = elementType.getIntOrFloatBitWidth() / 8;
  int64_t transBufferSize = bpe;
  SmallVector<int64_t> transShape(rank - 1, 1);
  for (int i = 1; i < rank; i++) {
    transBufferSize *= shapes[i];
    transShape[i - 1] = shapes[i];
  }
  auto ptrType = gcu::PtrType::get(buffertype.getContext(), elementType);
  Value ptr = rewriter.create<gcu::MemRefToPtrOp>(loc, ptrType, TransBuffers);
  MemRefType memType1D =
      MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());
  auto buffer1D = rewriter.create<gcu::PtrToMemRefOp>(loc, memType1D, ptr);
  auto bufferWithSpace = rewriter.create<memref::MemorySpaceCastOp>(
      loc,
      MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(), AffineMap{},
                      buffertype.getMemorySpace()),
      buffer1D);
  auto I8Offset = rewriter.create<arith::MulIOp>(
      loc, tagindex,
      rewriter.create<arith::ConstantIntOp>(loc, tagindex.getType(),
                                            transBufferSize));
  auto resultType = MemRefType::get(transShape, elementType, AffineMap{},
                                    buffertype.getMemorySpace());
  auto output = rewriter.create<memref::ViewOp>(
      loc, resultType, bufferWithSpace,
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                          I8Offset),
      ValueRange{});
  return output;
}

Operation *getBufferDealloc(triton::gcu::BufferAllocOp op) {
  auto users = op->getUsers();
  for (auto user : users) {
    if (isa_and_nonnull<triton::gcu::BufferDeallocOp>(user)) {
      return user;
    }
  }
  return nullptr;
}

void getPipelineAsyncResourceMaping(
    Operation *module, std::map<Operation *, Operation *> &asyncLoad2Tag,
    std::map<Operation *, Operation *> &asyncLoad2TransBuffers,
    llvm::DenseMap<Operation *, Value> &asyncLoad2Tagidex,
    std::map<Operation *, Operation *> &asyncWait2Tag) {
  int32_t pipelineResourceNumber = -1;
  std::map<Operation *, Operation *> shareAlloc2Tags;
  std::map<Operation *, Operation *> shareAlloc2TransBuffers;
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
    auto tag = builder.create<memref::AllocOp>(op.getLoc(), tagType);
    ModuleOp builtinModule = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gcu::getNumWarps(builtinModule);
    int rank = type.getShape().size();
    auto dealloc = getBufferDealloc(op);
    assert(dealloc != nullptr && "buffer alloc is not be dealloced");
    if (numWarps > 1) {
      tag->setAttr("gcu.share_tag", builder.getUnitAttr());
      if (rank > 2) {
        auto transBuffertype = MemRefType::get(
            type.getShape(), type.getElementType(), AffineMap{},
            IntegerAttr::get(IntegerType::get(type.getContext(), 64), 2));
        auto transBuffer =
            builder.create<memref::AllocOp>(op.getLoc(), transBuffertype);
        auto ip = builder.saveInsertionPoint();
        builder.setInsertionPoint(dealloc);
        builder.create<memref::DeallocOp>(op.getLoc(), transBuffer);
        builder.restoreInsertionPoint(ip);
        shareAlloc2TransBuffers[op.getOperation()] = transBuffer.getOperation();
      }
    } else {
      tag->setAttr("gcu.private_tag", builder.getUnitAttr());
      if (rank > 2) {
        auto transBuffertype =
            MemRefType::get(type.getShape(), type.getElementType());
        auto transBuffer =
            builder.create<memref::AllocOp>(op.getLoc(), transBuffertype);
        auto ip = builder.saveInsertionPoint();
        builder.setInsertionPoint(dealloc);
        builder.create<memref::DeallocOp>(op.getLoc(), transBuffer);
        builder.restoreInsertionPoint(ip);
        shareAlloc2TransBuffers[op.getOperation()] = transBuffer.getOperation();
        shareAlloc2TransBuffers[op.getOperation()] = transBuffer.getOperation();
      }
    }
    shareAlloc2Tags[op.getOperation()] = tag.getOperation();
  });

  auto getShareAlloc = [&](Operation *tokenDefineOp) {
    assert(isa<triton::gcu::AsyncLoadFromGlobalOp>(tokenDefineOp) &&
           " wait_op's tag should be a AsyncLoadFromGlobalOp op!");
    auto dstbuffer =
        dyn_cast<triton::gcu::AsyncLoadFromGlobalOp>(tokenDefineOp).getDstMem();
    auto bufferDefineOp = dstbuffer.getDefiningOp();

    Operation *shareAllocOp = nullptr;

    if (bufferDefineOp && isa<triton::gpu::MemDescIndexOp>(bufferDefineOp)) {
      auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(bufferDefineOp);
      shareAllocOp = indexOp.getSrc().getDefiningOp();
      if (!shareAllocOp || !isa<triton::gcu::BufferAllocOp>(shareAllocOp)) {
        assert(false &&
               " MemDescIndexOp's source should be a BufferAllocOp op!");
      }
    } else if (bufferDefineOp &&
               isa<triton::gpu::MemDescSubsliceOp>(bufferDefineOp)) {
      auto subView = dyn_cast<triton::gpu::MemDescSubsliceOp>(bufferDefineOp);
      shareAllocOp = subView.getSrc().getDefiningOp();
      if (!shareAllocOp || !isa<triton::gcu::BufferAllocOp>(shareAllocOp)) {
        assert(false &&
               " MemDescSubsliceOp's src should be a BufferAllocOp op!");
      }
    } else {
      assert(false && " AsyncLoadFromGlobalOp's dst should be a MemDescIndexOp "
                      "or MemDescSubsliceOp!");
    }

    return shareAllocOp;
  };

  module->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    llvm::TypeSwitch<mlir::Operation *>(operation)
        .Case<triton::gcu::AsyncLoadFromGlobalOp>(
            [&](triton::gcu::AsyncLoadFromGlobalOp load) {
              auto dstbuffer = load.getDstMem();
              auto defineOp = dstbuffer.getDefiningOp();

              if (defineOp && isa<triton::gpu::MemDescIndexOp>(defineOp)) {
                auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(defineOp);
                auto shareAllocOp = indexOp.getSrc().getDefiningOp();
                if (!shareAllocOp ||
                    !isa<triton::gcu::BufferAllocOp>(shareAllocOp)) {
                  assert(false &&
                         " MemDescIndexOp's source should be a BufferAllocOp!");
                }
                asyncLoad2Tag[operation] = shareAlloc2Tags[shareAllocOp];
                asyncLoad2TransBuffers[operation] =
                    shareAlloc2TransBuffers[shareAllocOp];
                asyncLoad2Tagidex[operation] = indexOp.getIndex();
              } else if (defineOp &&
                         isa<triton::gpu::MemDescSubsliceOp>(defineOp)) {
                auto subView =
                    dyn_cast<triton::gpu::MemDescSubsliceOp>(defineOp);
                auto shareAllocOp = subView.getSrc().getDefiningOp();
                if (!shareAllocOp ||
                    !isa<triton::gcu::BufferAllocOp>(shareAllocOp)) {
                  assert(false &&
                         " MemDescSubsliceOp's src should be a BufferAllocOp!");
                }
                asyncLoad2Tag[operation] = shareAlloc2Tags[shareAllocOp];
                asyncLoad2TransBuffers[operation] =
                    shareAlloc2TransBuffers[shareAllocOp];
                auto opOffsets = subView.getOffsets();
                OpBuilder builder(subView);
                SmallVector<Value> opOffsetVals;
                for (int offset : opOffsets) {
                  opOffsetVals.push_back(builder.create<arith::ConstantIntOp>(
                      subView.getLoc(), offset, 32));
                }
                asyncLoad2Tagidex[operation] = opOffsetVals[0];
              } else {
                assert(false && " AsyncLoadFromGlobalOp's dst should be a "
                                "MemDescIndexOp or MemDescSubsliceOp!");
              }
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
  matchAndRewrite(triton::gcu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto srcLayout =
        cast<triton::gpu::TensorOrMemDesc>(op.getSrc().getType()).getEncoding();
    auto dstLayout = dyn_cast<RankedTensorType>(op.getType()).getEncoding();
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto firstUser =
        userAnalysis.getFirstUser(op.getOperation()->getResults()[0]);
    triton::gcu::TagInfo tag;
    if (firstUser.first != nullptr) {
      tag = pTagPool.trygGetAsyncTagInfo(op);
    } else {
      tag = pTagPool.getSyncTagInfo(op);
    }
    if (tag.isAsync()) {
      pTagPool.setMap(firstUser.first, tag);
    }
    // share to Distributed
    if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
      // copy to local
      ModuleOp builtinModule = op->getParentOfType<ModuleOp>();
      int numWarps = triton::gcu::getNumWarps(builtinModule);
      if (numWarps > 1) {
        auto output = loadFromSharedMem(
            rewriter, tag, op.getResult().getType(), adaptor.getSrc(), false,
            lastUser, firstUser, userAnalysis, replaced2Origin);
        leaveTritionOp(rewriter, op.getOperation());
        rewriter.replaceOp(op, output);
        return success();
      } else {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
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

struct TTMemDescSubsliceOpLowering
    : SharedConversionPattern<triton::gpu::MemDescSubsliceOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescSubsliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto sourceType = dyn_cast<MemRefType>(src.getType());
    auto sourceRank = sourceType.getRank();
    auto [strides, offset] = sourceType.getStridesAndOffset();
    (void)offset;
    auto opOffsets = op.getOffsets();
    SmallVector<Value> opOffsetVals;
    for (long int offset : opOffsets) {
      opOffsetVals.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, offset, 32));
    }
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
          loc, opOffsetVals[0].getType(), strides[i]));
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
        rewriter.create<arith::ConstantIntOp>(loc, opOffsetVals[0].getType(),
                                              bpe));
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
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTMemDescIndexOpLowering
    : SharedConversionPattern<triton::gpu::MemDescIndexOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto sourceType = dyn_cast<MemRefType>(src.getType());
    auto [strides, offset] = sourceType.getStridesAndOffset();
    (void)offset;

    auto indexValue = adaptor.getIndex();

    auto strideValue = rewriter.create<arith::ConstantIntOp>(
        loc, indexValue.getType(), strides[0]);
    auto finalOffsetValue =
        rewriter.create<arith::MulIOp>(loc, indexValue, strideValue);

    auto elemType = resultType.getElementType();
    auto bpe = elemType.getIntOrFloatBitWidth() / 8;
    auto elementType = resultType.getElementType();

    int64_t size = 1;
    for (int i = 0; i < sourceType.getRank(); i++) {
      size *= sourceType.getShape()[i];
    }

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
        rewriter.create<arith::ConstantIntOp>(loc, indexValue.getType(), bpe));
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
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTAsyncLoadFromGlobalOpLowering
    : SharedConversionPattern<triton::gcu::AsyncLoadFromGlobalOp> {
  using SharedConversionPattern::SharedConversionPattern;

  std::map<Operation *, Operation *> &asyncLoad2Tag;
  llvm::DenseMap<Operation *, Value> &asyncLoad2Tagidex;
  std::map<Operation *, Operation *> asyncLoad2TransBuffers;

  explicit TTAsyncLoadFromGlobalOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      triton::gcu::FirstLastUserAnalysis &userAnalysis,
      std::map<Operation *, Operation *> &replaced2Origin,
      triton::gcu::PrivateDTETagPool &pTagPool,
      std::map<Operation *, Operation *> &inAsyncLoad2Tags,
      llvm::DenseMap<Operation *, Value> &inAsyncLoad2Tagidex,
      std::map<Operation *, Operation *> asyncLoad2TransBuffers)
      : SharedConversionPattern<triton::gcu::AsyncLoadFromGlobalOp>(
            converter, ctx, userAnalysis, replaced2Origin, pTagPool),
        asyncLoad2Tag(inAsyncLoad2Tags), asyncLoad2Tagidex(inAsyncLoad2Tagidex),
        asyncLoad2TransBuffers(asyncLoad2TransBuffers) {}

  LogicalResult
  matchAndRewrite(triton::gcu::AsyncLoadFromGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    bool isPrologueLoad = false;
    if (op.getOperation()->getAttr("Prologue_stage_idex")) {
      isPrologueLoad = true;
    }
    auto loc = op.getLoc();
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
    assert((asyncLoad2Tag.find(op.getOperation()) != asyncLoad2Tag.end()) &&
           "AsyncLoadFromGlobalOp had no mapping tags !!!");
    assert((asyncLoad2Tagidex.find(op.getOperation()) !=
            asyncLoad2Tagidex.end()) &&
           "AsyncLoadFromGlobalOp had no mapping tagindx !!!");
    Value outBroadcast = adaptor.getDstMem();
    if (rank > 1) {
      auto itTrans = asyncLoad2TransBuffers.find(op.getOperation());
      assert((itTrans != asyncLoad2TransBuffers.end()) &&
             "can't find transeBuffers!!!");
      outBroadcast =
          getTransBuffer(rewriter, loc, itTrans->second->getResult(0),
                         asyncLoad2Tagidex[op.getOperation()]);
    }
    if (isPrologueLoad == true) {
      int32_t prologueIdx = dyn_cast<IntegerAttr>(op.getOperation()->getAttr(
                                                      "Prologue_stage_idex"))
                                .getInt();
      // get range from for
      Operation *forUser = nullptr;
      int32_t userNumber = 0;
      for (Operation *user : op.getOperation()->getUsers()) {
        userNumber++;
        if (isa<scf::ForOp>(user)) {
          forUser = user;
        }
      }
      if (forUser == nullptr || userNumber > 2) {
        op.dump();
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
            auto isThread0 = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq,
                builder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
            auto defaultValue =
                op.getDefaultValue()
                    ? adaptor.getDefaultValue()
                    : triton::gcu::createConstantZero(builder, loc, elemType);
            auto tagIdx = builder
                              .create<arith::IndexCastOp>(
                                  loc, builder.getIndexType(),
                                  asyncLoad2Tagidex[op.getOperation()])
                              .getResult();
            builder.create<scf::IfOp>(
                loc, isThread0, [&](OpBuilder &builder, Location loc) {
                  auto attr = op.getOperation()->getAttr(triton::gcu::kLoadEx);
                  if (attr && cast<BoolAttr>(attr).getValue()) {
                    ConfigGcuLoadEx(
                        rewriter, loc, pTagPool, adaptor.getDstMem(),
                        outBroadcast, op.getOperation(), outputType,
                        adaptor.getPtr(), adaptor.getStrides(),
                        adaptor.getShape(), defaultValue,
                        triton::gcu::TagInfo(
                            asyncLoad2Tag[op.getOperation()]->getResult(0),
                            tagIdx, true),
                        true);
                  } else {
                    ConfigGcuLoad(
                        builder, loc, pTagPool, adaptor.getDstMem(),
                        outBroadcast, op.getOperation(), outputType,
                        adaptor.getPtr(), adaptor.getStrides(),
                        adaptor.getShape(), defaultValue,
                        triton::gcu::TagInfo(
                            asyncLoad2Tag[op.getOperation()]->getResult(0),
                            tagIdx, true),
                        true);
                  }
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::YieldOp>(loc);
          });
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, asyncLoad2Tagidex[op.getOperation()]);
      return success();
    }
    // to avoid share momeory race
    rewriter.create<gpu::BarrierOp>(loc);
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
    auto defaultValue =
        op.getDefaultValue()
            ? adaptor.getDefaultValue()
            : triton::gcu::createConstantZero(rewriter, loc, elemType);
    auto tagIdx =
        rewriter
            .create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                        asyncLoad2Tagidex[op.getOperation()])
            .getResult();
    rewriter.create<scf::IfOp>(
        loc, isThread0, [&](OpBuilder &builder, Location loc) {
          auto attr = op.getOperation()->getAttr(triton::gcu::kLoadEx);
          if (attr && cast<BoolAttr>(attr).getValue()) {
            ConfigGcuLoadEx(builder, loc, pTagPool, adaptor.getDstMem(),
                            outBroadcast, op.getOperation(), outputType,
                            adaptor.getPtr(), adaptor.getStrides(),
                            adaptor.getShape(), defaultValue,
                            triton::gcu::TagInfo(
                                asyncLoad2Tag[op.getOperation()]->getResult(0),
                                tagIdx, true),
                            true);
          } else {
            ConfigGcuLoad(builder, loc, pTagPool, adaptor.getDstMem(),
                          outBroadcast, op.getOperation(), outputType,
                          adaptor.getPtr(), adaptor.getStrides(),
                          adaptor.getShape(), defaultValue,
                          triton::gcu::TagInfo(
                              asyncLoad2Tag[op.getOperation()]->getResult(0),
                              tagIdx, true),
                          true);
          }
          builder.create<scf::YieldOp>(loc);
        });
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, asyncLoad2Tagidex[op.getOperation()]);
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
      triton::gcu::PrivateDTETagPool &pTagPool,
      std::map<Operation *, Operation *> &inAsyncWait2Tag)
      : SharedConversionPattern<triton::gcu::AsyncWaitOp>(
            converter, ctx, userAnalysis, replaced2Origin, pTagPool),
        asyncWait2Tag(inAsyncWait2Tag) {}

  LogicalResult
  matchAndRewrite(triton::gcu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    assert((asyncWait2Tag.find(op.getOperation()) != asyncWait2Tag.end()) &&
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
          WaitGcuLoadStore(
              builder, loc,
              triton::gcu::TagInfo(
                  asyncWait2Tag[op.getOperation()]->getResult(0), tagIdx, true),
              zero);
          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<gpu::BarrierOp>(loc);
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, adaptor.getAsyncToken()[0]);
    return success();
  }
};
#endif
