/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "ppu/include/TritonPPUGPUToLLVM/TIXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::LLVM::PPU::lowerLdStMatrix;

constexpr int kPtrBitWidth = 64;
struct ConvertLayoutOpSwizzlingConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const ppu::TargetInfo &targetInfo;

  explicit ConvertLayoutOpSwizzlingConversion(LLVMTypeConverter &typeConverter,
                                              const ppu::TargetInfo &targetInfo,
                                              PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = toLinearLayout(srcTy);
    LinearLayout dstLayout = toLinearLayout(dstTy);

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kReg = str_attr("register");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
    if (!llvm::is_contained(dims, kBlock) &&
        cvtNeedsSharedMemory(srcTy, dstTy)) {
      auto loc = op.getLoc();
      // Remove the kBlock dimension from the layout as it's the identity in the
      // cvt
      srcLayout = srcLayout.sublayout({kReg, kLane, kWarp},
                                      to_vector(srcLayout.getOutDimNames()));
      dstLayout = dstLayout.sublayout({kReg, kLane, kWarp},
                                      to_vector(dstLayout.getOutDimNames()));

      auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
      auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase);

      Value result =
          packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }

  SmallVector<Value> transferWithinBlockSwizzling(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals =
          transferWithinBlockSwizzling(loc, rewriter, srcLayout, dstLayout,
                                       newInVals, llvmElemTyPtr, smemBase);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      // Upcast to i8
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v);
      }
      return outVals;
    }

    // Remove broadcasting in src
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferWithinBlockSwizzling(loc, rewriter, prmtSrc, dstLayout,
                                          newInVals, llvmElemTy, smemBase);
    }

    // Remove broadcasting in dst
    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, prmtDst, inVals, llvmElemTy, smemBase);
      return broadcastAs(outVals, dstLayout);
    }

    // At this point we have a type that's at least 8-bit
    // and we don't have broadcasting in the registers
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth);
    auto [smem, instr] =
        optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
    auto [idxSrc, idxDst] = instr;

    // Extract reps from smem
    auto kReg = str_attr("register");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = dstLayout.invertAndCompose(smem);

    // The permutation exists by construction of the reps dimension in
    // optimalSwizzling
    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    // Remove the reps and flatten into offset
    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    storeCvt = storeCvt.reshapeOuts({{kOffset, storeCvt.getTotalOutDimSize()}});
    loadCvt = loadCvt.reshapeOuts({{kOffset, loadCvt.getTotalOutDimSize()}});

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;
    auto noPaddingOffset = [](Value v) { return v; };
    bool isWarpSync = mlir::isCvtWarpSync(srcLayout, dstLayout);
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        targetInfo.barrier(loc, rewriter, isWarpSync);

      auto tileInVals =
          to_vector(ArrayRef(permutedInVals).slice(i * tileSize, tileSize));
      // Store
      // idxSrc 0: st.shared, idxSrc 1: stmatrix, idxSrc 2: stmatrix.trans
      if (idxSrc == 0) {
        lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                        noPaddingOffset, affineOffset, maskSpanAffineOffset,
                        rewriter, targetInfo);
      } else {
        assert(idxSrc == 1 || idxSrc == 2);
        bool transpose = idxSrc == 2;
        auto result = lowerLdStMatrix(
            loc, storeCvt, transpose, tileInVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      targetInfo.barrier(loc, rewriter, isWarpSync);
      // Load
      SmallVector<Value> tileOutVals;
      // idxDst 0: ld.shared, idxDst 1: ldmatrix, idxDst 2: ldmatrix.trans
      if (idxDst == 0) {
        tileOutVals = lowerLdStShared(
            loc, ctx, loadCvt, {}, llvmElemTy, smemBase, noPaddingOffset,
            affineOffset, maskSpanAffineOffset, rewriter, targetInfo);
      } else {
        assert(idxDst == 1 || idxDst == 2);
        bool transpose = idxDst == 2;
        auto result = lowerLdStMatrix(
            loc, loadCvt, transpose, tileOutVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      llvm::append_range(outVals, tileOutVals);
    }

    // Undo the permLoad used to divideRight
    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }

  LogicalResult
  transferWithinBlockSwizzling(ConvertLayoutOp op, Value src,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    // Remove the kBlock dimension from the layout as it's the identity in the
    // cvt
    auto srcLayout = toLinearLayout(srcTy);
    auto dstLayout = toLinearLayout(dstTy);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    srcLayout = srcLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(srcLayout.getOutDimNames()));
    dstLayout = dstLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(dstLayout.getOutDimNames()));

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = unpackLLElements(loc, src, rewriter);
    auto outVals = transferWithinBlockSwizzling(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase);

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(const LLVMTypeConverter &typeConverter,
                            const ppu::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (isa<PPUMmaEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    return failure();
  }

private:
  void convertMmaV1ToDotOperand(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto dotOperandLayout =
        dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());

    // The layout conversion between MMA and dotOperand (i.e. ChainedDot
    // optimization) can be implemented in several ways:
    //
    // 1.`SharedMemory`: convert layout through shared memory store & load.(only
    //    used in Triton <=3.2.0).
    // 2.`DotOperand0`: apply a transformation matrix 16x16x16 fp16 to operand0.
    // 3.`DotOperand1`: apply a transformation matrix 16x16x16 fp16 to operand1.
    //    This optimization is equivalent to using the `ppu.ldmatrix`
    //    instruction directly, which requires to update the linear layout of
    //    `ppu.ldmatrix` in `lowerPPULdMatrix`.

    // ChainedDot optimization by `DotOperand1`:
    if (dotOperandLayout.getIsChained()) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return;
    }

    // ChainedDot optimization by `DotOperand0`:
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    unsigned elems = getTotalElemsPerThread(srcTy);
    Type elemTy = this->getTypeConverter()->convertType(srcTy.getElementType());
    auto elemSize = elemTy.getIntOrFloatBitWidth();
    unsigned vecSize = std::max<unsigned>(32 / elemSize, 1);
    Type vecTy = vec_ty(elemTy, vecSize);

    // for the destination type, we need to pack values together
    // so they can be consumed by tensor core operations
    SmallVector<Value> vecVals;
    for (unsigned i = 0; i < elems; i += vecSize) {
      Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
      for (unsigned j = 0; j < vecSize; j++)
        packed = b.insert_element(vecTy, packed, vals[i + j], b.i32_val(j));
      vecVals.push_back(b.bitcast(packed, i32_ty));
    }

    auto mmaLayout =
        dyn_cast<triton::gpu::PPUMmaEncodingAttr>(srcTy.getEncoding());
    assert(mmaLayout.getVecSize() == 1);
    unsigned numMmaRets = 4;

    // F16 or BF16
    auto srcElemTy = srcTy.getElementType();
    bool isF16 = srcElemTy.isF16();
    auto mmaInstrTix =
        isF16 ? "ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16"
              : "ppu.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16";
    auto mmaTy = LLVM::LLVMStructType::getLiteral(
        op.getContext(), SmallVector<Type>(numMmaRets, vec_ty(srcElemTy, 2)));
    SmallVector<Value> reorderedVals;
    // mma is generated from FP32_FP16_FP16_FP32, and casted to FP16 results
    for (unsigned i = 0; i < vecVals.size(); i += numMmaRets) {
      mlir::triton::ppu::TIXBuilder builder;
      auto retArgs = builder.newListOperand(numMmaRets, isF16 ? "=r" : "=f");
      auto aArgs = builder.newListOperand({
          {vecVals[i], "r"},
          {vecVals[i + 1], "r"},
          {vecVals[i + 2], "r"},
          {vecVals[i + 3], "r"},
      });
      auto Zero = b.i32_val(0);
      auto cArgs = builder.newListOperand(
          {{Zero, "r"}, {Zero, "r"}, {Zero, "r"}, {Zero, "r"}});
      auto thread = getThreadId(rewriter, loc);
      Value lane = b.urem(thread, b.i32_val(32));
      Value high_phase = b.lshr(lane, b.i32_val(4));
      Value high_phase_t = b.icmp_eq(high_phase, b.i32_val(1));
      Value high_phase_f = b.icmp_eq(high_phase, b.i32_val(0));
      Value inner_idx = b.and_(lane, b.i32_val(0xf));
      Value row = b.lshr(inner_idx, b.i32_val(2));
      Value col = b.and_(inner_idx, b.i32_val(3));
      Value rce = b.icmp_eq(row, col);
      Value Value1 = b.and_(high_phase_f, rce);
      Value Value2 = b.and_(high_phase_t, rce);
      Value ValueL = isF16 ? b.i32_val(0x3c00) : b.i32_val(0x3f80);
      Value ValueH = isF16 ? b.i32_val(0x3c000000) : b.i32_val(0x3f800000);
      Value vreg0 = b.i32_val(0);
      Value vreg1 = b.i32_val(0);
      Value vreg2 = b.i32_val(0);
      Value vreg3 = b.i32_val(0);

      vreg0 = b.select(Value1, ValueL, vreg0);
      vreg0 = b.select(Value2, ValueH, vreg0);

      vreg3 = vreg0;
      auto bArgs = builder.newListOperand(
          {{vreg0, "r"}, {vreg1, "r"}, {vreg2, "r"}, {vreg3, "r"}});

      auto &mma = *builder.create(mmaInstrTix);
      mma(retArgs, aArgs, bArgs, cArgs);
      Value mmaOut = builder.launch(rewriter, loc, mmaTy);
      Type mmaElemTy =
          cast<LLVM::LLVMStructType>(mmaOut.getType()).getBody()[0];
      // for (int i = 0; i < numMmaRets; ++i) {
      //   reorderedVals.push_back(b.bitcast(b.extract_val(mmaElemTy, mmaOut,
      //   i), i32_ty));
      // }
      for (unsigned i = 0; i < numMmaRets; i++) {
        Value vecVal = b.bitcast(b.extract_val(mmaElemTy, mmaOut, i), vecTy);
        for (unsigned j = 0; j < vecSize; j++) {
          Value elemVal = b.extract_element(elemTy, vecVal, b.i32_val(j));
          reorderedVals.push_back(elemVal);
        }
      }
    }
    Value view =
        packLLElements(loc, getTypeConverter(), reorderedVals, rewriter, dstTy);
    rewriter.replaceOp(op, view);
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    if (mlir::LLVM::PPU::matchMmaV1AndDotOperandLayout(srcTy, dstTy)) {
      convertMmaV1ToDotOperand(op, adaptor, rewriter);
      return success();
    }
    return failure();
  }

private:
  const ppu::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::ppu::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Give this convertLayoutOpConversion a higher benefit as it only matches
  // optimized or cross CTA cases
  patterns.add<ConvertLayoutOpConversion, ConvertLayoutOpSwizzlingConversion>(
      typeConverter, targetInfo, benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
