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

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {
struct TritonLoadOpLowering : SharedConversionPattern<triton::LoadOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, loadOp.getOperation());
    if (pTagPool.isExistInMap(loadOp.getOperation())) {
      pTagPool.releaseMap(loadOp.getOperation());
    }
    auto loc = loadOp.getLoc();
    assert(!(isa<triton::PointerType>(loadOp.getPtr().getType()) &&
             isa<RankedTensorType>(
                 dyn_cast<triton::PointerType>(loadOp.getPtr().getType())
                     .getPointeeType())));

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto loadType = loadOp.getType();
    // tensor
    if (isa<TensorType>(loadType)) {
      auto lastUser =
          userAnalysis.getLastUser(loadOp.getOperation()->getResults()[0]);
      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(loadOp.getType()));
      auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, resultType);
      ///////////////////////////////////////////////////////////////////////////'
      auto elementType = resultType.getElementType();
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      unsigned vectorLength = oaccSizeInBytes / bpe;
      auto totalNumElems = triton::gcu::getTotalElemsPerThread(loadType);
      auto end = rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
      triton::gcu::TritonGCUBuilder tarBuild(loc, rewriter);
      SmallVector<Value> init_args;
      auto args_ptr = tarBuild.tarAddr(adaptor.getPtr());
      auto args_v = tarBuild.tarAddr(output);
      auto masks_ptr_ty = gcu::PtrType::get(getContext(), rewriter.getI1Type());
      Value other = triton::gcu::createConstantZero(
          rewriter, loc, resultType.getElementType());
      if (adaptor.getOther()) {
        auto otherOp = loadOp.getOther().getDefiningOp();
        if (auto elementwiseFusionOp =
                dyn_cast_or_null<triton::gcu::ElementwiseFusionRegionOp>(
                    otherOp)) {
          unsigned resultIndex =
              cast<OpResult>(loadOp.getOther()).getResultNumber();
          auto yieldOp = elementwiseFusionOp.getRegion().back().getTerminator();
          if (resultIndex < yieldOp->getNumOperands() &&
              dyn_cast_or_null<arith::ConstantOp>(
                  yieldOp->getOperand(resultIndex).getDefiningOp())) {
            otherOp = yieldOp->getOperand(resultIndex).getDefiningOp();
          }
        }

        if (auto cstOther = dyn_cast_or_null<arith::ConstantOp>(otherOp)) {
          if (auto splatAttr =
                  llvm::dyn_cast<SplatElementsAttr>(cstOther.getValue())) {
            mlir::Type elementType = splatAttr.getElementType();
            if (elementType.isIntOrIndex()) {
              other = rewriter.create<arith::ConstantIntOp>(
                  loc, elementType,
                  splatAttr.getSplatValue<APInt>().getSExtValue());
            } else if (elementType.isBF16() || elementType.isF16() ||
                       elementType.isTF32() || elementType.isF32() ||
                       elementType.isF64() ||
                       llvm::isa<Float8E4M3B11FNUZType>(elementType) ||
                       llvm::isa<Float8E4M3FNUZType>(elementType) ||
                       llvm::isa<Float8E5M2FNUZType>(elementType) ||
                       llvm::isa<Float8E4M3FNType>(elementType) ||
                       llvm::isa<Float8E5M2Type>(elementType)) {
              // float v =
              //   splatAttr.getSplatValue<APFloat>()
              other = rewriter.create<arith::ConstantFloatOp>(
                  loc, llvm::cast<mlir::FloatType>(elementType),
                  splatAttr.getSplatValue<APFloat>());
            }
          } else {
            llvm_unreachable(
                "Unsupported constant other op in TritonLoadOpLowering");
            return failure();
          }
        } else {
          llvm_unreachable("Unsupported other op in TritonLoadOpLowering");
          return failure();
        }
      }
      Value masks_ptr = nullptr;
      if (adaptor.getMask())
        masks_ptr = rewriter.create<gcu::MemRefToPtrOp>(loc, masks_ptr_ty,
                                                        adaptor.getMask());
      auto vt_v = VectorType::get(ArrayRef<int64_t>{vectorLength}, elementType);
      auto vt_other = rewriter.create<vector::BroadcastOp>(loc, vt_v, other);
      init_args.push_back(args_v);
      init_args.push_back(args_ptr);
      if (masks_ptr)
        init_args.push_back(masks_ptr);
      auto tarStride = tarBuild.tarValue(oaccSizeInBytes);
      rewriter.create<scf::ForOp>(
          loc, zero, end, step, init_args,
          [&](OpBuilder &builder, Location loc, Value iter,
              ValueRange iterArgs) {
            SmallVector<Value> args(iterArgs);
            auto num = builder.create<arith::IndexCastOp>(
                loc, builder.getI32Type(),
                builder.create<arith::MinSIOp>(
                    loc, step, builder.create<arith::SubIOp>(loc, end, iter)));
            auto v = tarBuild.tarGather(vt_v, args[1], num, vt_other,
                                        args.size() > 2 ? args[2] : masks_ptr);
            tarBuild.tarStore(v, args[0], tarStride);
            if (args.size() > 2) {
              args[2] = builder.create<gcu::IntToPtrOp>(
                  loc, masks_ptr_ty,
                  builder.create<arith::AddIOp>(
                      loc, builder.create<gcu::PtrToIntOp>(loc, args[2]),
                      builder.create<arith::ConstantIntOp>(loc, vectorLength,
                                                           64)));
            }
            builder.create<scf::YieldOp>(loc, args);
          });
      leaveTritionOp(rewriter, loadOp.getOperation());
      rewriter.replaceOp(loadOp, output);
      return success();
    } else {
      // scalar
      auto mask =
          adaptor.getMask()
              ? adaptor.getMask()
              : rewriter.create<arith::ConstantIntOp>(loc, 1, 1).getResult();
      auto other = adaptor.getOther() ? adaptor.getOther()
                                      : triton::gcu::createConstantZero(
                                            rewriter, loc, loadOp.getType());
      auto elemType =
          dyn_cast<gcu::PtrType>(adaptor.getPtr().getType()).getElementType();
      auto memType =
          MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
      auto buffer =
          rewriter.create<gcu::PtrToMemRefOp>(loc, memType, adaptor.getPtr());

      auto scalarV = rewriter.create<scf::IfOp>(
          loc, mask,
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::YieldOp>(
                loc, ValueRange{builder.create<memref::LoadOp>(
                         loc, buffer, ValueRange{zero})});
          },
          [&](OpBuilder &builder, Location loc) {
            builder.create<scf::YieldOp>(loc, ValueRange{other});
          });

      leaveTritionOp(rewriter, loadOp.getOperation());
      rewriter.replaceOp(loadOp, scalarV);
      return success();
    }
  }
};

struct TritonStoreOpLowering : SharedConversionPattern<triton::StoreOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, storeOp.getOperation());
    if (pTagPool.isExistInMap(storeOp.getOperation())) {
      pTagPool.releaseMap(storeOp.getOperation());
    }
    auto loc = storeOp.getLoc();
    assert(!(isa<triton::PointerType>(storeOp.getPtr().getType()) &&
             isa<RankedTensorType>(
                 dyn_cast<triton::PointerType>(storeOp.getPtr().getType())
                     .getPointeeType())));

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // tensor
    if (isa<TensorType>(storeOp.getPtr().getType())) {
      auto values = adaptor.getValue();
      auto valueType = dyn_cast<MemRefType>(values.getType());
      auto elementType = valueType.getElementType();
      auto bpe = mlir::triton::gcu::getBpe(elementType);
      unsigned vectorLength = oaccSizeInBytes / bpe;
      auto totalNumElems =
          triton::gcu::getTotalElemsPerThread(storeOp.getPtr().getType());
      auto end = rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, vectorLength);
      triton::gcu::TritonGCUBuilder tarBuild(loc, rewriter);
      SmallVector<Value> init_args;
      auto args_ptr = tarBuild.tarAddr(adaptor.getPtr());
      auto args_v = tarBuild.tarAddr(values);
      auto masks_ptr_ty = gcu::PtrType::get(getContext(), rewriter.getI1Type());
      Value masks_ptr = nullptr;
      if (adaptor.getMask())
        masks_ptr = rewriter.create<gcu::MemRefToPtrOp>(loc, masks_ptr_ty,
                                                        adaptor.getMask());
      auto vt_v = VectorType::get(ArrayRef<int64_t>{vectorLength}, elementType);
      init_args.push_back(args_ptr);
      init_args.push_back(args_v);
      if (masks_ptr)
        init_args.push_back(masks_ptr);
      auto tarStride = tarBuild.tarValue(oaccSizeInBytes);
      rewriter.create<scf::ForOp>(
          loc, zero, end, step, init_args,
          [&](OpBuilder &builder, Location loc, Value iter,
              ValueRange iterArgs) {
            SmallVector<Value> args(iterArgs);
            auto v = tarBuild.tarLoad(vt_v, args[1], tarStride);
            auto num = builder.create<arith::IndexCastOp>(
                loc, builder.getI32Type(),
                builder.create<arith::MinSIOp>(
                    loc, step, builder.create<arith::SubIOp>(loc, end, iter)));
            if (args.size() > 2) {
              tarBuild.tarScatter(args[0], v, num, args[2]);
              args[2] = builder.create<gcu::IntToPtrOp>(
                  loc, masks_ptr_ty,
                  builder.create<arith::AddIOp>(
                      loc, builder.create<gcu::PtrToIntOp>(loc, args[2]),
                      builder.create<arith::ConstantIntOp>(loc, vectorLength,
                                                           64)));
            } else {
              tarBuild.tarScatter(args[0], v, num, masks_ptr);
            }
            builder.create<scf::YieldOp>(loc, args);
          });

      leaveTritionOp(rewriter, storeOp.getOperation());
      rewriter.eraseOp(storeOp);
      return success();
    } else {
      // scalar
      auto elemType =
          dyn_cast<gcu::PtrType>(adaptor.getPtr().getType()).getElementType();
      auto memType =
          MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
      auto buffer =
          rewriter.create<gcu::PtrToMemRefOp>(loc, memType, adaptor.getPtr());
      auto mask =
          adaptor.getMask()
              ? adaptor.getMask()
              : rewriter.create<arith::ConstantIntOp>(loc, 1, 1).getResult();
      auto isThread0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
      mask = rewriter.create<arith::AndIOp>(loc, mask, isThread0);
      rewriter.create<scf::IfOp>(
          loc, mask, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::StoreOp>(loc, adaptor.getValue(), buffer,
                                            ValueRange{zero});
            builder.create<scf::YieldOp>(loc);
          });
      leaveTritionOp(rewriter, storeOp.getOperation());
      rewriter.eraseOp(storeOp);
      return success();
    }
  }
};
} // namespace

void mlir::triton::populateLoadStoreOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<TritonLoadOpLowering, TritonStoreOpLowering>(
      converter, patterns.getContext(), userAnalysis, replaced2Origin,
      pTagPool);
}
