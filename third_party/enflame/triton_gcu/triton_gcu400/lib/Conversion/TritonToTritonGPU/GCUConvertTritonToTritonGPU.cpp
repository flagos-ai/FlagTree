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
#include <numeric>
#include <utility>

#include "GCUTritonGPUConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "Utils/TritonVersionCompat.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

#ifdef ENABLE_TRITON_DISTRIBUTED
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"
#endif

#ifdef ENABLE_TLE
#include "tle/dialect/include/IR/Dialect.h"
#endif

namespace mlir {
#define GEN_PASS_DECL_GCUCONVERTTRITONTOTRITONGPUPASS
#define GEN_PASS_DEF_GCUCONVERTTRITONTOTRITONGPUPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

//===----------------------------------------------------------------------===//
// Generic pattern: convert result types via the type converter
//===----------------------------------------------------------------------===//

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
#ifdef ENABLE_TLE
    if (failed(
            this->getTypeConverter()->convertTypes(op->getResults(), retTypes)))
#else
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
#endif
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ArithConstantPattern
//===----------------------------------------------------------------------===//

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType =
#ifdef ENABLE_TLE
        getTypeConverter()->convertType(op.getResult());
#else
        getTypeConverter()->convertType(op.getType());
#endif
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (isa<RankedTensorType>(retShapedType)) {
      assert(value && "expected a dense elements attribute");
      value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arith patterns
//===----------------------------------------------------------------------===//

void populateArithPatternsAndLegality(GCUTritonGPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      GCUTritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, GenericOpPattern<arith::AddFOp>,
      GenericOpPattern<arith::SubFOp>, GenericOpPattern<arith::MaximumFOp>,
      GenericOpPattern<arith::MaxNumFOp>, GenericOpPattern<arith::MaxSIOp>,
      GenericOpPattern<arith::MaxUIOp>, GenericOpPattern<arith::MinimumFOp>,
      GenericOpPattern<arith::MinNumFOp>, GenericOpPattern<arith::MinSIOp>,
      GenericOpPattern<arith::MinUIOp>, GenericOpPattern<arith::MulFOp>,
      GenericOpPattern<arith::DivFOp>, GenericOpPattern<arith::RemFOp>,
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      GenericOpPattern<arith::SelectOp>, GenericOpPattern<arith::TruncIOp>,
      GenericOpPattern<arith::TruncFOp>, GenericOpPattern<arith::ExtUIOp>,
      GenericOpPattern<arith::ExtSIOp>, GenericOpPattern<arith::ExtFOp>,
      GenericOpPattern<arith::SIToFPOp>, GenericOpPattern<arith::FPToSIOp>,
      GenericOpPattern<arith::FPToUIOp>, GenericOpPattern<arith::UIToFPOp>>(
      typeConverter, context);
}

//===----------------------------------------------------------------------===//
// Math patterns
//===----------------------------------------------------------------------===//

void populateMathPatternsAndLegality(GCUTritonGPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     GCUTritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
               GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
               GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//===----------------------------------------------------------------------===//
// Triton-specific patterns
//===----------------------------------------------------------------------===//

struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  bool preserveOrder;

  TritonExpandDimsPattern(GCUTritonGPUTypeConverter &typeConverter,
                          MLIRContext *context, bool preserveOrder)
      : OpConversionPattern(typeConverter, context),
        preserveOrder(preserveOrder) {}

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType argType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = cast<triton::gpu::BlockedEncodingAttr>(_argEncoding);
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    auto newRank = retShape.size();
    auto retSizePerThread = llvm::to_vector(argEncoding.getSizePerThread());
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    auto retThreadsPerWarp = to_vector(argEncoding.getThreadsPerWarp());
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    auto retWarpsPerCTA = to_vector(argEncoding.getWarpsPerCTA());
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(newRank);
    if (preserveOrder) {
      for (unsigned i = 0; i < argEncoding.getOrder().size(); ++i) {
        unsigned o = argEncoding.getOrder()[i];
        retOrder[i] = o >= op.getAxis() ? o + 1 : o;
      }
      retOrder[newRank - 1] = op.getAxis();
    } else {
      std::iota(retOrder.begin(), retOrder.end(), 0);
    }
#if TRITON_VERSION == 35
    auto argCTALayout = argEncoding.getCTALayout();
    auto retCTAsPerCGA = insertOne(argCTALayout.getCTAsPerCGA(), op.getAxis());
    auto retCTASplitNum =
        insertOne(argCTALayout.getCTASplitNum(), op.getAxis());
    auto retCTAOrder = insertOrder(argCTALayout.getCTAOrder(), op.getAxis());
    auto retCTALayout = triton::gpu::CTALayoutAttr::get(
        getContext(), retCTAsPerCGA, retCTASplitNum, retCTAOrder);
#else
    auto ctaLl = triton_gcu::compat::getCGALayout(argEncoding).getLinearLayout();
    auto kBlock = *ctaLl.getInDimNames().begin();
    auto *ctx = kBlock.getContext();
    auto newDim = standardOutDimNames(ctx, newRank)[newRank - 1];
    ctaLl *= LinearLayout::identity1D(1, kBlock, newDim);
    auto newOrder = to_vector(llvm::seq<int32_t>(newRank));
    for (int i = newRank - 1; i >= static_cast<int>(op.getAxis()) + 1; --i) {
      std::swap(newOrder[i], newOrder[i - 1]);
    }
    ctaLl = transposeLinearLayout(ctaLl, newOrder);
    auto retCTALayout = triton_gcu::compat::getCGALayoutFromLL(ctx, std::move(ctaLl));
#endif

    triton::gpu::BlockedEncodingAttr retEncoding =
        triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder, retCTALayout);
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = argType.cloneWithEncoding(newArgEncoding);
    auto newSrc = triton::gpu::ConvertLayoutOp::create(
        rewriter, op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

#if TRITON_VERSION == 35
private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
#endif
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<GCUTritonGPUTypeConverter>();
#ifdef ENABLE_TLE
    int numWarps = triton::gpu::lookupNumWarps(op);
#else
    int numWarps = typeConverter->getNumWarps();
#endif
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();
    auto rank = origShape.size();
    SmallVector<unsigned> retSizePerThread(rank, 1);
    auto numElements = product<int64_t>(origShape);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }
    retSizePerThread[rank - 1] = std::min(
        retSizePerThread[rank - 1], static_cast<unsigned>(origShape[rank - 1]));
    retSizePerThread[rank - 2] = std::min(
        retSizePerThread[rank - 2], static_cast<unsigned>(origShape[rank - 2]));

    SmallVector<unsigned> retOrder(rank);
    for (unsigned i = 0; i < rank; ++i)
      retOrder[i] = rank - 1 - i;
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    RankedTensorType retType = origType.cloneWithEncoding(dEncoding);
    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = cast<RankedTensorType>(adaptor.getB().getType());
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType = aType.cloneWithEncoding(encoding);
      a = triton::gpu::ConvertLayoutOp::create(rewriter, a.getLoc(), dstType,
                                               a);
    }
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(bEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType = bType.cloneWithEncoding(encoding);
      b = triton::gpu::ConvertLayoutOp::create(rewriter, b.getLoc(), dstType,
                                               b);
    }
    c = triton::gpu::ConvertLayoutOp::create(rewriter, c.getLoc(), retType, c);

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::DotOp>(
                      op, retType, a, b, c, adaptor.getInputPrecision(),
                      adaptor.getMaxNumImpreciseAcc()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto retType = cast<RankedTensorType>(
#ifdef ENABLE_TLE
        this->getTypeConverter()->convertType(op.getResult()));
#else
        this->getTypeConverter()->convertType(op.getType()));
#endif
    auto retEncoding =
        cast<triton::gpu::BlockedEncodingAttr>(retType.getEncoding());
    auto lhsType = adaptor.getLhs().getType();
    auto rhsType = adaptor.getRhs().getType();
    auto lhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(lhsType);
    auto rhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(rhsType);
    auto retTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(retType);
    auto retOrder = retEncoding.getOrder();
    auto retThreadsPerWarp = retEncoding.getThreadsPerWarp();
    auto retWarpsPerCTA = retEncoding.getWarpsPerCTA();
    auto newRetTotalElemsPerThread =
        nextPowOf2(lhsTotalElemsPerThread + rhsTotalElemsPerThread);
    auto newRetSizePerThread = llvm::to_vector(retEncoding.getSizePerThread());
    newRetSizePerThread[retOrder[0]] *=
        newRetTotalElemsPerThread / retTotalElemsPerThread;
    triton::gpu::BlockedEncodingAttr newRetEncoding =
        triton::gpu::BlockedEncodingAttr::get(
            getContext(), newRetSizePerThread, retThreadsPerWarp,
            retWarpsPerCTA, retOrder, triton_gcu::compat::getCGALayout(retEncoding));
    auto newRetType = retType.cloneWithEncoding(newRetEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::CatOp>(
                      op, newRetType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    if (!srcTy.getEncoding())
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<TransOp>(op, src, op.getOrder()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = op.getType().cloneWithEncoding(srcEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = triton::ReduceOp::create(
        rewriter, op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());
    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newScan =
        triton::ScanOp::create(rewriter, op.getLoc(), adaptor.getOperands(),
                               adaptor.getAxis(), op.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());
    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    TypeConverter::SignatureConversion result(op.getNumArguments());
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (!newOp.getBody().empty())
      rewriter.applySignatureConversion(&newOp.getBody().front(), result,
                                        converter);
    return success();
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::JoinOp>(
                      op, adaptor.getLhs(), adaptor.getRhs()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    int rank = srcEnc.getOrder().size();
    auto typeConverter = getTypeConverter<GCUTritonGPUTypeConverter>();

    if (!srcEnc || srcEnc.getSizePerThread().back() != 2 ||
        static_cast<int>(srcEnc.getOrder().front()) != rank - 1) {
      auto defaultEnc = getDefaultBlockedEncoding(
          getContext(),
          cast<RankedTensorType>(op.getResult(0).getType()).getShape(),
#ifdef ENABLE_TLE
          triton::gpu::lookupNumWarps(op), typeConverter->getThreadsPerWarp(),
#else
          typeConverter->getNumWarps(), typeConverter->getThreadsPerWarp(),
#endif
          typeConverter->getNumCTAs());

      auto append = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res(vals);
        res.push_back(val);
        return res;
      };
      auto prepend = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res;
        res.push_back(val);
        res.append(vals.begin(), vals.end());
        return res;
      };

#if TRITON_VERSION == 35
      srcEnc = BlockedEncodingAttr::get(
          getContext(), append(defaultEnc.getSizePerThread(), 2),
          append(defaultEnc.getThreadsPerWarp(), 1),
          append(defaultEnc.getWarpsPerCTA(), 1),
          prepend(defaultEnc.getOrder(), rank - 1),
          CTALayoutAttr::get(getContext(),
                             append(defaultEnc.getCTAsPerCGA(), 1),
                             append(defaultEnc.getCTASplitNum(), 1),
                             prepend(defaultEnc.getCTAOrder(), rank - 1)));
#else
      auto layout = triton_gcu::compat::getCGALayout(defaultEnc).getLinearLayout();
      auto kBlock = StringAttr::get(getContext(), "block");
      auto newDim = standardOutDimNames(getContext(), rank)[rank - 1];
      layout *= LinearLayout::identity1D(1, kBlock, newDim);
      srcEnc = BlockedEncodingAttr::get(
          getContext(), append(defaultEnc.getSizePerThread(), 2),
          append(defaultEnc.getThreadsPerWarp(), 1),
          append(defaultEnc.getWarpsPerCTA(), 1),
          prepend(defaultEnc.getOrder(), rank - 1),
          triton_gcu::compat::getCGALayoutFromLL(getContext(), layout));
#endif
      srcTy = srcTy.cloneWithEncoding(srcEnc);
      src = ConvertLayoutOp::create(rewriter, op.getLoc(), srcTy, src);
    }

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::SplitOp>(op, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonMapElementwisePattern
    : public OpConversionPattern<triton::MapElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MapElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    SmallVector<Type> resultTys;
    auto err = converter->convertTypes(op.getResults().getType(), resultTys);
    if (failed(err)) {
      return err;
    }

    auto newMapOp = triton::MapElementwiseOp::create(
        rewriter, op.getLoc(), resultTys, adaptor.getOperands(), op.getPack());
    addNamedAttrs(newMapOp, adaptor.getAttributes());

    auto &newScalarOp = newMapOp.getScalarOp();
    rewriter.cloneRegionBefore(op.getScalarOp(), newScalarOp,
                               newScalarOp.end());
    rewriter.replaceOp(op, newMapOp.getResult());
    return success();
  }
};

class TritonCallOpPattern : public OpConversionPattern<triton::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<triton::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class TritonReturnOpPattern : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateTritonPatterns(GCUTritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns,
                            bool hasReduceOps) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<
      // clang-format off
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>,
      GenericOpPattern<triton::ReshapeOp>,
      GenericOpPattern<triton::BitcastOp>,
      GenericOpPattern<triton::FpToFpOp>,
      GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>,
      GenericOpPattern<triton::SplatOp>,
      GenericOpPattern<triton::UnsplatOp>,
      GenericOpPattern<triton::AddPtrOp>,
      TritonBroadcastPattern,
      TritonCatPattern,
      TritonJoinOpPattern,
      TritonSplitOpPattern,
      GenericOpPattern<triton::ClampFOp>,
      GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>,
      GenericOpPattern<triton::MulhiUIOp>,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>,
      TritonReducePattern,
      GenericOpPattern<triton::ReduceReturnOp>,
      TritonScanPattern,
      GenericOpPattern<triton::ScanReturnOp>,
      GenericOpPattern<triton::MakeRangeOp>,
      TritonTransPattern,
      TritonDotPattern,
      TritonMapElementwisePattern,
      GatherScatterOpPattern<triton::DescriptorGatherOp>,
      GatherScatterOpPattern<triton::DescriptorScatterOp>,
      GenericOpPattern<triton::LoadOp>,
      GenericOpPattern<triton::StoreOp>,
      GenericOpPattern<triton::HistogramOp>,
      GenericOpPattern<triton::GatherOp>,
      GenericOpPattern<triton::ExternElementwiseOp>,
      GenericOpPattern<triton::PrintOp>,
      GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::AtomicCASOp>,
      GenericOpPattern<triton::AtomicRMWOp>,
      GenericOpPattern<triton::DescriptorLoadOp>,
      GenericOpPattern<triton::DescriptorStoreOp>,
      GenericOpPattern<triton::DescriptorReduceOp>,
      GenericOpPattern<triton::DotScaledOp>,
      TritonCallOpPattern,
      TritonReturnOpPattern,
      TritonFuncOpPattern
      // clang-format on
      >(typeConverter, context);
  patterns.insert<TritonExpandDimsPattern>(typeConverter, context,
                                           hasReduceOps);
}


//===----------------------------------------------------------------------===//
// Tensor dialect patterns
//===----------------------------------------------------------------------===//

void populateTensorPatterns(GCUTritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<tensor::ExtractOp>,
               GenericOpPattern<tensor::InsertOp>>(typeConverter, context);
}

//===----------------------------------------------------------------------===//
// Distributed patterns
//===----------------------------------------------------------------------===//

#ifdef ENABLE_TRITON_DISTRIBUTED
struct SIMTExecRegionPattern
    : public OpConversionPattern<triton::simt::SIMTExecRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::simt::SIMTExecRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = cast<triton::simt::SIMTExecRegionOp>(
        rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter())))
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    newOp->setOperands(adaptor.getOperands());
    for (auto [result, operand] :
         llvm::zip(newOp.getResults(), adaptor.getOperands()))
      result.setType(operand.getType());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

void populateDistributedPatterns(GCUTritonGPUTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<triton::distributed::WaitOp>>(typeConverter,
                                                              context);
  patterns.add<GenericOpPattern<triton::distributed::ConsumeTokenOp>>(
      typeConverter, context);
  patterns.add<SIMTExecRegionPattern>(typeConverter, context);
  patterns.add<GenericOpPattern<triton::simt::BlockYieldOp>>(typeConverter,
                                                             context);
}
#endif

//===----------------------------------------------------------------------===//
// SCF patterns
//===----------------------------------------------------------------------===//

struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter())))
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    newOp->setOperands(adaptor.getOperands());
    SmallVector<Type> newResultTypes;
#ifdef ENABLE_TLE
    for (Value result : op.getResults()) {
      Type newType = typeConverter->convertType(result);
#else
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
#endif
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
#ifdef ENABLE_TLE
    for (Value result : op.getResults()) {
      Type newType = typeConverter->convertType(result);
#else
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
#endif
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();
    auto newOp = scf::WhileOp::create(rewriter, op.getLoc(), newResultTypes,
                                      adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(GCUTritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

//===----------------------------------------------------------------------===//
// CF patterns
//===----------------------------------------------------------------------===//

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

void populateCFPatterns(GCUTritonGPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}

//===----------------------------------------------------------------------===//
// TLE patterns
//===----------------------------------------------------------------------===//

#ifdef ENABLE_TLE
class TleDSLRegionOpPattern
    : public OpConversionPattern<triton::tle::DSLRegionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::tle::DSLRegionOp op,
                  triton::tle::DSLRegionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.cloneWithoutRegions<triton::tle::DSLRegionOp>(op);
    Region &body = op.getBody(), &newBody = newOp.getBody();
    rewriter.inlineRegionBefore(body, newBody, newBody.end());

    if (failed(rewriter.convertRegionTypes(&newBody, *getTypeConverter())))
      return rewriter.notifyMatchFailure(op, "could not convert body types");

    newOp->setOperands(adaptor.getOperands());
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults()))
      newResult.setType(getTypeConverter()->convertType(oldResult));

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class TleExtractTileOpPattern
    : public OpConversionPattern<triton::tle::ExtractTileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::tle::ExtractTileOp op,
                  triton::tle::ExtractTileOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<RankedTensorType>(adaptor.getSrc().getType());
    if (!srcType)
      return op.emitError("source must be a ranked tensor");
    auto srcEnc = srcType.getEncoding();
    if (!srcEnc)
      return op.emitError("source tensor must have encoding attribute");

    Type retType = op.getType().cloneWithEncoding(srcEnc);
    auto newOp = rewriter.replaceOpWithNewOp<triton::tle::ExtractTileOp>(
        op, retType, adaptor.getSrc(), adaptor.getIndex());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class TleInsertTileOpPattern
    : public OpConversionPattern<triton::tle::InsertTileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::tle::InsertTileOp op,
                  triton::tle::InsertTileOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<RankedTensorType>(adaptor.getSrc().getType());
    if (!srcType)
      return op.emitError("source must be a ranked tensor");
    auto srcEnc = srcType.getEncoding();
    if (!srcEnc)
      return op.emitError("source tensor must have encoding attribute");

    Type retType = op.getType().cloneWithEncoding(srcEnc);
    auto newOp = rewriter.replaceOpWithNewOp<triton::tle::InsertTileOp>(
        op, retType, adaptor.getSrc(), adaptor.getTile(), adaptor.getIndex());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

void populateTlePatterns(GCUTritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TleDSLRegionOpPattern, TleExtractTileOpPattern,
               TleInsertTileOpPattern,
               GenericOpPattern<triton::tle::LocalPointersOp>,
               GenericOpPattern<triton::tle::RemotePointersOp>,
               GenericOpPattern<triton::tle::ExclusiveCumsumOp>,
               GenericOpPattern<triton::tle::DistributedBarrierOp>,
               GenericOpPattern<triton::tle::YieldOp>,
               GenericOpPattern<triton::tle::ExtractAllocatedPtrOp>,
               GenericOpPattern<triton::tle::ExtractAlignedPtrOp>,
               GenericOpPattern<triton::tle::ExtractOffsetOp>,
               GenericOpPattern<triton::tle::ExtractSizesOp>,
               GenericOpPattern<triton::tle::ExtractStridesOp>,
               GenericOpPattern<triton::tle::ExtractPtrOp>,
               GenericOpPattern<triton::tle::PackOp>>(typeConverter, context);
}

#endif

//===----------------------------------------------------------------------===//
// Reduce-aware order analysis
//===----------------------------------------------------------------------===//

// Scan the module for ops that perform cross-thread communication along an
// axis: tt.reduce, tt.scan, and tle.exclusive_cumsum. All of these benefit
// from NOT being the most-contiguous dimension in the layout, because that
// would force expensive DTE transfers during the operation.
static llvm::SmallDenseMap<unsigned, unsigned>
collectReduceAxesWithFreq(ModuleOp mod) {
  llvm::SmallDenseMap<unsigned, unsigned> axisFreq;
  mod.walk([&](triton::ReduceOp reduceOp) {
    ++axisFreq[static_cast<unsigned>(reduceOp.getAxis())];
  });
  mod.walk([&](triton::ScanOp scanOp) {
    ++axisFreq[static_cast<unsigned>(scanOp.getAxis())];
  });
  mod.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tle.exclusive_cumsum") {
      if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis"))
        ++axisFreq[static_cast<unsigned>(axisAttr.getInt())];
    }
  });
  return axisFreq;
}

static SmallVector<unsigned>
buildReduceAwareOrder(unsigned rank,
                      const llvm::SmallDenseMap<unsigned, unsigned> &axisFreq) {
  SmallVector<unsigned> nonReduceDims;
  SmallVector<std::pair<unsigned, unsigned>> reduceDimsWithFreq;
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
    unsigned dim = static_cast<unsigned>(i);
    auto it = axisFreq.find(dim);
    if (it != axisFreq.end())
      reduceDimsWithFreq.push_back({dim, it->second});
    else
      nonReduceDims.push_back(dim);
  }
  // Sort reduce dims by frequency ascending: less frequent = higher priority.
  llvm::sort(reduceDimsWithFreq, [](const std::pair<unsigned, unsigned> &a,
                                    const std::pair<unsigned, unsigned> &b) {
    if (a.second != b.second)
      return a.second < b.second;
    return a.first > b.first;
  });
  SmallVector<unsigned> order;
  order.append(nonReduceDims.begin(), nonReduceDims.end());
  for (auto &kv : reduceDimsWithFreq)
    order.push_back(kv.first);
  return order;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

class GCUConvertTritonToTritonGPU
    : public mlir::impl::GCUConvertTritonToTritonGPUPassBase<
          GCUConvertTritonToTritonGPU> {
public:
  using GCUConvertTritonToTritonGPUPassBase::
      GCUConvertTritonToTritonGPUPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    auto axisFreq = collectReduceAxesWithFreq(mod);
    unsigned maxRank = 0;
    mod.walk([&](Operation *op) {
      for (auto type : op->getResultTypes()) {
        if (auto tensorType = dyn_cast<RankedTensorType>(type))
          maxRank =
              std::max(maxRank, static_cast<unsigned>(tensorType.getRank()));
      }
      for (auto type : op->getOperandTypes()) {
        if (auto tensorType = dyn_cast<RankedTensorType>(type))
          maxRank =
              std::max(maxRank, static_cast<unsigned>(tensorType.getRank()));
      }
    });
    if (maxRank == 0)
      maxRank = 1;

    bool orderIncompatible = false;
    mod.walk([&](Operation *op) -> WalkResult {
      if (isa<triton::DotOp>(op) || isa<triton::ReshapeOp>(op) ||
          isa<triton::SplitOp>(op) || isa<triton::JoinOp>(op)) {
        orderIncompatible = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    SmallVector<unsigned> defaultOrder;
    if (orderIncompatible || axisFreq.empty()) {
      axisFreq.clear();
      defaultOrder.resize(maxRank);
      std::iota(defaultOrder.begin(), defaultOrder.end(), 0);
      std::reverse(defaultOrder.begin(), defaultOrder.end());
    } else {
      defaultOrder = buildReduceAwareOrder(maxRank, axisFreq);
    }

    // Step 2: Build type converter and conversion target.
    // Module attributes must be set BEFORE conversion so that
    // lookupNumWarps / lookupThreadsPerWarp can resolve context from
    // the module for values that aren't inside WS partition regions.
    Builder b(&getContext());
    mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(AttrNumThreadsPerWarp, b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    if (!this->target.getValue().empty())
      mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

    // Worker functions (e.g. TLE warp-specialization workers) may carry a
    // different ttg.num-warps attribute than the kernel-level numWarps (e.g.
    // producer-default: kernel num_warps=1, consumer worker num_warps=4).
    //
    // The MLIR conversion framework's reconciliation phase uses
    // convertType(Type) (context-unaware) which can only read the module-level
    // ttg.num-warps attribute. If the module-level value doesn't match the
    // function being processed, incorrect encodings are generated.
    //
    // To handle this generically, we run applyPartialConversion per-function,
    // setting the module-level ttg.num-warps to match each function's
    // attribute before processing it.
    {
      // Collect all functions in the module.
      SmallVector<triton::FuncOp, 4> funcs;
      mod.walk([&](triton::FuncOp func) { funcs.push_back(func); });

      for (triton::FuncOp func : funcs) {
        // Determine the numWarps for this function.
        int funcNumWarps = numWarps;
        if (auto attr = func->getAttrOfType<IntegerAttr>(AttrNumWarpsName))
          funcNumWarps = attr.getInt();

        // Set module-level ttg.num-warps to match this function so that
        // convertType(Type) (used by the framework's reconciliation) uses
        // the correct numWarps.
        mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(funcNumWarps));

        // Build type converter and target for this function's numWarps.
        GCUTritonGPUTypeConverter typeConverter(context, funcNumWarps,
                                                threadsPerWarp, numCTAs,
                                                defaultOrder, axisFreq);
        GCUTritonGPUConversionTarget target(*context, typeConverter);

        // Populate patterns (must be done per type converter).
        RewritePatternSet patterns(context);
        populateArithPatternsAndLegality(typeConverter, patterns, target);
        populateMathPatternsAndLegality(typeConverter, patterns, target);
        populateTritonPatterns(typeConverter, patterns, !axisFreq.empty());
#ifdef ENABLE_TRITON_DISTRIBUTED
        populateDistributedPatterns(typeConverter, patterns);
#endif
        populateTensorPatterns(typeConverter, patterns);
        populateSCFPatterns(typeConverter, patterns);
        populateCFPatterns(typeConverter, patterns);
        patterns.insert<GenericOpPattern<ub::PoisonOp>>(typeConverter, context);
#ifdef ENABLE_TLE
        populateTlePatterns(typeConverter, patterns);
#endif

        if (failed(applyPartialConversion(
                ArrayRef<Operation *>{func.getOperation()}, target,
                std::move(patterns))))
          return signalPassFailure();
      }

      // Restore module-level numWarps to the original value.
      mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    }
  }
};

} // namespace
