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
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

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
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
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
    Type retType = getTypeConverter()->convertType(op.getType());
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
  using OpConversionPattern::OpConversionPattern;

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
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    auto ctaLl = argEncoding.getCTALayout().getLinearLayout();
    auto kBlock = *ctaLl.getInDimNames().begin();
    auto *ctx = kBlock.getContext();
    auto newDim = standardOutDimNames(ctx, newRank)[newRank - 1];
    ctaLl *= LinearLayout::identity1D(1, kBlock, newDim);
    auto newOrder = to_vector(llvm::seq<int32_t>(newRank));
    for (int i = newRank - 1; i >= static_cast<int>(op.getAxis()) + 1; --i) {
      std::swap(newOrder[i], newOrder[i - 1]);
    }
    ctaLl = transposeLinearLayout(ctaLl, newOrder);
    auto retCTALayout = CTAEncodingAttr::get(ctx, std::move(ctaLl));

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
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<GCUTritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
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
        this->getTypeConverter()->convertType(op.getType()));
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
            retWarpsPerCTA, retOrder, retEncoding.getCTALayout());
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
          typeConverter->getNumWarps(), typeConverter->getThreadsPerWarp(),
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

      auto layout = defaultEnc.getCTALayout().getLinearLayout();
      auto kBlock = StringAttr::get(getContext(), "block");
      auto newDim = standardOutDimNames(getContext(), rank)[rank - 1];
      layout *= LinearLayout::identity1D(1, kBlock, newDim);
      srcEnc = BlockedEncodingAttr::get(
          getContext(), append(defaultEnc.getSizePerThread(), 2),
          append(defaultEnc.getThreadsPerWarp(), 1),
          append(defaultEnc.getWarpsPerCTA(), 1),
          prepend(defaultEnc.getOrder(), rank - 1),
          CTAEncodingAttr::get(getContext(), layout));
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
                            RewritePatternSet &patterns) {
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
      TritonExpandDimsPattern,
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
}

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
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
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
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
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
// Reduce-aware order analysis
//===----------------------------------------------------------------------===//

// Scan the module for tt.reduce ops and collect reduce axes with frequency.
// Returns a map: axis -> occurrence count.
static llvm::SmallDenseMap<unsigned, unsigned>
collectReduceAxesWithFreq(ModuleOp mod) {
  llvm::SmallDenseMap<unsigned, unsigned> axisFreq;
  mod.walk([&](triton::ReduceOp reduceOp) {
    ++axisFreq[static_cast<unsigned>(reduceOp.getAxis())];
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
  for (auto &[dim, freq] : reduceDimsWithFreq)
    order.push_back(dim);
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
    auto originalOrder = buildReduceAwareOrder(maxRank, axisFreq);
    bool hasDot = false;
    bool orderIncompatible = false;
    mod.walk([&](Operation *op) -> WalkResult {
      if (isa<triton::DotOp>(op)) {
        hasDot = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!hasDot && !axisFreq.empty()) {
      auto *inferInterface =
          context->getOrLoadDialect<triton::gpu::TritonGPUDialect>()
              ->getRegisteredInterface<triton::DialectInferLayoutInterface>();

      if (!inferInterface) {
        orderIncompatible = true;
      } else {
        auto trialEnc = [&](ArrayRef<int64_t> shape) {
          return getBlockedEncodingWithOrder(context, shape, originalOrder,
                                             axisFreq, numWarps, threadsPerWarp,
                                             numCTAs);
        };
        auto isBlockedResult = [](Attribute enc) {
          return isa<triton::gpu::BlockedEncodingAttr>(enc);
        };

        mod.walk([&](Operation *op) -> WalkResult {
          if (orderIncompatible)
            return WalkResult::interrupt();

          if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op)) {
            if (reshapeOp.getAllowReorder())
              return WalkResult::advance();
            ArrayRef<int64_t> srcShape =
                reshapeOp.getSrc().getType().getShape();
            ArrayRef<int64_t> dstShape = reshapeOp.getType().getShape();

            auto srcEnc = trialEnc(srcShape);
            Attribute inferredDst;
            if (failed(inferInterface->inferReshapeOpEncoding(
                    srcShape, srcEnc, dstShape, inferredDst, std::nullopt)) ||
                !isBlockedResult(inferredDst)) {
              orderIncompatible = true;
              return WalkResult::interrupt();
            }

            auto dstEnc = trialEnc(dstShape);
            Attribute inferredSrc;
            if (failed(inferInterface->inferReshapeOpEncoding(
                    dstShape, dstEnc, srcShape, inferredSrc, std::nullopt)) ||
                !isBlockedResult(inferredSrc)) {
              orderIncompatible = true;
              return WalkResult::interrupt();
            }
          }

          if (auto splitOp = dyn_cast<triton::SplitOp>(op)) {
            auto inputShape = splitOp.getSrc().getType().getShape();
            auto enc = trialEnc(inputShape);
            Attribute inferredDst;
            if (failed(inferInterface->inferSplitOpEncoding(
                    enc, inferredDst, inputShape, std::nullopt)) ||
                !isBlockedResult(inferredDst)) {
              orderIncompatible = true;
              return WalkResult::interrupt();
            }
          }

          if (auto joinOp = dyn_cast<triton::JoinOp>(op)) {
            auto resultShape = joinOp.getType().getShape();
            auto enc = trialEnc(resultShape);
            Attribute inferredDst;
            if (failed(inferInterface->inferSplitOpEncoding(
                    enc, inferredDst, resultShape, std::nullopt)) ||
                !isBlockedResult(inferredDst)) {
              orderIncompatible = true;
              return WalkResult::interrupt();
            }
          }

          return WalkResult::advance();
        });
      }
    }

    SmallVector<unsigned> defaultOrder;
    if (hasDot || orderIncompatible || axisFreq.empty()) {
      defaultOrder.resize(maxRank);
      std::iota(defaultOrder.begin(), defaultOrder.end(), 0);
      std::reverse(defaultOrder.begin(), defaultOrder.end());
    } else {
      defaultOrder = buildReduceAwareOrder(maxRank, axisFreq);
    }

    // Step 2: Build type converter and conversion target.
    GCUTritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                            numCTAs, defaultOrder, axisFreq);
    GCUTritonGPUConversionTarget target(*context, typeConverter);

    // Step 3: Populate rewrite patterns.
    RewritePatternSet patterns(context);
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns);
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);
    patterns.insert<GenericOpPattern<ub::PoisonOp>>(typeConverter, context);

    // Step 4: Set module attributes.
    Builder b(&getContext());
    mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(AttrNumThreadsPerWarp, b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    if (!this->target.getValue().empty())
      mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
