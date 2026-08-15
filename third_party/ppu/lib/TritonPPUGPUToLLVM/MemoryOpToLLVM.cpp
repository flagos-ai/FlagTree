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

#include "Dialect/PPUGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonPPUGPUToLLVM/AIUUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

namespace SharedToDotOperandPPUAIUV1 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread,
                    ArrayRef<unsigned> aiuLoad);
}

namespace SharedToDotOperandPPUAIUV2 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread,
                    ArrayRef<unsigned> aiuLoad);
}

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::ppu;
using namespace mlir::LLVM::PPU;

LogicalResult lowerLdStMatrix(
    Location loc, const LinearLayout &regLayout, MemDescType memDescType,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    SharedMemoryObject smemObj, ConversionPatternRewriter &rewriter,
    const ppu::TargetInfo &targetInfo, const LLVMTypeConverter *typeConverter) {
  bool isStore = !vals.empty();

  // Remove broadcasting from regLayout
  auto removeBroadcast = actionRemoveBroadcastedRegs(regLayout);
  if (!removeBroadcast.isIdentity()) {
    if (isStore) {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      vals = removeBroadcast.apply(vals);
      return lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                             rewriter, targetInfo, typeConverter);
    } else {
      auto newRegLayout = removeBroadcast.apply(regLayout);
      auto result =
          lowerLdStMatrix(loc, newRegLayout, memDescType, vals, smemObj,
                          rewriter, targetInfo, typeConverter);
      if (succeeded(result)) {
        vals = broadcastAs(vals, regLayout);
      }
      return result;
    }
  }
  if (isa<PaddedSharedEncodingAttr>(memDescType.getEncoding())) {
    return failure();
  }
  auto memLayout = toLinearLayout(memDescType);
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto kBlock = StringAttr::get(loc.getContext(), "block");
  auto maybeSublayout = cvt.quotient({kBlock});
  if (!maybeSublayout) {
    return failure();
  }
  cvt = maybeSublayout.value();
  auto smemBase = smemObj.getBase();
  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescType);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(memDescType);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
  for (bool transpose : {false, true}) {
    auto result = LLVM::PPU::lowerLdStMatrix(
        loc, cvt, transpose, vals, smemBase, affineOffset, maskSpanAffineOffset,
        llvmElemTy, rewriter, targetInfo);
    if (succeeded(result)) {
      return result;
    }
  }
  return failure();
}

LogicalResult lowerPPULdMatrix(
    Location loc, RankedTensorType tensorTy, MemDescType memDescType,
    bool transpose, Value &src, // Output for ldmatrix
    Value smemBase, Type llvmElemTy, ConversionPatternRewriter &rewriter,
    const ppu::TargetInfo &targetInfo, const LLVMTypeConverter *typeConverter,
    std::pair<size_t, Type> *const llvmOpCount = nullptr) {
  // Lower load via ppu.ldmatrix
  if (!targetInfo.supportLdMatrix())
    return failure();

  bool isPPU0010 = false;
  bool isPPU0015 = false;
  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(tensorTy.getEncoding());
  if (dotEnc) {
    auto mmaEncoding =
        dyn_cast<triton::gpu::PPUMmaEncodingAttr>(dotEnc.getParent());
    if (mmaEncoding && (mmaEncoding.isPPU0010() || mmaEncoding.isPPU0015())) {
      isPPU0010 = mmaEncoding.isPPU0010();
      isPPU0015 = mmaEncoding.isPPU0015();
    }
  }
  if (!isPPU0010 && !isPPU0015)
    return failure();

  auto rank = tensorTy.getRank();
  auto kOrder = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
  auto nonKOrder = dotEnc.getOpIdx() == 0 ? rank - 2 : rank - 1;
  auto bitwidth = tensorTy.getElementTypeBitWidth();
  auto shape = tensorTy.getShape();

  // Limitation: check kWidth * bitwidth
  if (dotEnc.getKWidth() * bitwidth != 32)
    return failure();

  // Limitation: Only support 2d matrices now but we should
  // be able to support 3D minor changes
  if (rank > 2)
    return failure();

  // Limitation: Minimum tile size
  if (bitwidth == 8) {
    if (shape[kOrder] < 32 || shape[nonKOrder] < 16)
      return failure();
  } else {
    if (shape[kOrder] < (8 * 16 / bitwidth) || shape[nonKOrder] < 8)
      return failure();
  }

  assert(llvmOpCount == nullptr && "NYI");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = tensorTy.getContext();
  auto regL = toLinearLayout(tensorTy.getShape(), tensorTy.getEncoding());
  auto memL = toLinearLayout(memDescType.getShape(), memDescType.getEncoding());
  auto cvt = minimalCvtLayout(memDescType, tensorTy);

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kBlock = S("block");
  auto kOffset = S("offset");
  auto smemPtrTy = ptr_ty(ctx, 3);
  // In the transpose case, consecutive elements are not stored contiguously
  // so we cannot split an fp32
  // We could support bitwidth == 8, but it'd be a rather weird layout
  // so we don't do that for now
  // if ((!transpose && bitwidth > 32) || (transpose && bitwidth != 16))

  // support bitwidth == 8 on PPU
  if ((!transpose && bitwidth > 32) ||
      (transpose && bitwidth != 16 && bitwidth != 8))
    return failure();

  bool Opb8bLdmatrix =
      isPPU0010 && bitwidth == 8 && dotEnc.getOpIdx() == 1 && transpose;
  if (bitwidth == 8 && transpose && !Opb8bLdmatrix)
    return failure();

  // Inter block stmatrix is not supported
  if (cvt.hasInDim(kBlock))
    return failure();

  auto srcVals = SmallVector<Value>{};

  // Remove broadcasting on the register dimension
  auto removeBroadcast = actionRemoveBroadcastedRegs(cvt);
  cvt = removeBroadcast.apply(cvt);

  std::optional<ColumnAction> maybePermutation;
  LinearLayout tile;
  if (!transpose) {
    tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
           LinearLayout::identity1D(4, kLane, kOffset);

    // Find if there is a register permutation that allows us to divideLeft
    // We need to pass the map from regs to offsets, as is cvt
    // maybePermutation = regPermForDivideLeft(cvt, tile);
    maybePermutation = regPermForDivide(cvt, tile, /*left=*/true);
    if (!maybePermutation.has_value()) {
      return failure();
    }
    auto permutation = maybePermutation.value();
    // Check if the action indeed allows us to divideLeft
    cvt = permutation.apply(cvt);
  }

  LinearLayout reps;
  if (!transpose) {
    auto maybeQuot = divideLeft(cvt, tile);
    if (!maybeQuot.has_value()) {
      return failure();
    }
    reps = zerosLike(tile) * maybeQuot.value();
  } else {
    // Division does not quite work here. To define this properly, we would need
    // to define a different multiplication that does:
    // A *' B = [[0, A], [B, 0]] and define leftDivision for it
    // We do it ad-hoc for now, as I beleive there's not much demand for this op
    // outside of this lowering.

    // We implement leftDivision as above for B = identity1D(8, kLane, kOffset)
    // Divisibility in the sense above is the same as regular divisibility
    // You need to see that the tile A is a sublayout of the matrix, and that
    // it has zeros above it and to its right.

    // In particular, offsets lanes 4, 8, 16 map to offsets 1, 2, 4...
    const auto &laneBases = cvt.getBases().find(kLane)->second;
    for (int i = 0; i < 3; ++i) {
      if (laneBases[i + 2][0] != (1 << i))
        return failure();
    }
    // ... and no other basis should depend on 1, 2, 4
    // Note that this gives us the usual alignment condition, but we have
    // translated it to checking that the matrix to the left of A is all zeros
    for (auto dim : cvt.getInDimNames()) {
      const auto &bases = cvt.getBases().find(dim)->second;
      for (auto [i, basis] : llvm::enumerate(bases)) {
        if (dim == kLane && i >= 2)
          continue;
        if (basis[0] & 0b111)
          return failure();
      }
    }

    // Hack: We are not going to use in the rest of the function reps[kLane][2:]
    // so we don't need to zero them out
    reps = cvt;
  }

  // We must have at least 2 register elements to use stmatrix.trans
  if (transpose && reps.getInDimSizeLog2(kReg) < llvm::Log2_32(32 / bitwidth)) {
    return failure();
  }

  // Choose up to 4 packs of 32-bit elements indexed by the next (at most) two
  // bases as the vectorisation factor. We don't consider the basis of the tile
  // for vectorisation so we substract them
  auto vec = std::min<int32_t>(2, reps.getInDimSizeLog2(kReg) -
                                      llvm::Log2_32(32 / bitwidth));

  // Map from kReg, kLane, kWarp to beginning of each tile
  assert(reps.getOutDimSize(kOffset) == cvt.getOutDimSize(kOffset));

  LinearLayout addrLayout = choosePPULdMatrixLayout(dotEnc, shape, transpose,
                                                    bitwidth, Opb8bLdmatrix);
  LinearLayout sharedLayout =
      triton::gpu::toLinearLayout(shape, memDescType.getEncoding());
  LinearLayout regToSharedLayout = addrLayout.invertAndCompose(sharedLayout);
  std::optional<int32_t> maxVecElems = 8 * 16 / bitwidth;
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value regBase = applyLinearLayout(loc, rewriter, regToSharedLayout,
                                    {{kReg, b.i32_val(0)},
                                     {kLane, laneId},
                                     {kWarp, warpId},
                                     {kBlock, b.i32_val(0)}})[0]
                      .second;

  // Elements per op
  auto nVecs = 1 << vec;
  if (isPPU0010 && Opb8bLdmatrix)
    nVecs = 2;
  auto elemsPerVec = 32 / bitwidth;
  auto step = nVecs * elemsPerVec;
  for (int i = 0; i < cvt.getInDimSize(kReg); i += step) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    Value offset = b.xor_(regBase, b.i32_val(regIdx));
    auto vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset,
                         LLVM::GEPNoWrapFlags::inbounds);
    Type packedTy = vec_ty(llvmElemTy, 32 / bitwidth);
    Type matTy = nVecs == 1
                     ? i32_ty
                     : static_cast<Type>(LLVM::LLVMStructType::getLiteral(
                           ctx, SmallVector<Type>(nVecs, i32_ty)));
    Value res = rewriter
                    .create<triton::ppugpu::PPULoadMatrixOp>(
                        loc, matTy, vecAddr,
                        /*needTrans=*/transpose, Opb8bLdmatrix)
                    .getResult();

    // Extract result into srcVals
    bool needExchange = false;
    if (isPPU0015 && dotEnc && dotEnc.getOpIdx() == (transpose ? 0 : 1)) {
      needExchange = true;
    }
    if (needExchange) {
      auto resultType = cast<LLVM::LLVMStructType>(res.getType());
      assert(resultType.getBody().size() == 4 && "Unexpected vector size");
      SmallVector<Value> elemsI32;
      elemsI32.push_back(b.extract_val(i32_ty, res, 0));
      elemsI32.push_back(b.extract_val(i32_ty, res, 2));
      elemsI32.push_back(b.extract_val(i32_ty, res, 1));
      elemsI32.push_back(b.extract_val(i32_ty, res, 3));
      for (int j = 0; j < 4; j++) {
        Value output = b.bitcast(elemsI32[j], vec_ty(llvmElemTy, elemsPerVec));
        for (int k = 0; k < elemsPerVec; k++) {
          srcVals.push_back(
              b.extract_element(llvmElemTy, output, b.i32_val(k)));
        }
      }
    } else {
      for (int j = 0; j < nVecs; j++) {
        Value output = nVecs == 1 ? res : b.extract_val(i32_ty, res, j);
        output = b.bitcast(output, vec_ty(llvmElemTy, elemsPerVec));
        for (int k = 0; k < elemsPerVec; k++) {
          srcVals.push_back(
              b.extract_element(llvmElemTy, output, b.i32_val(k)));
        }
      }
    }
  }

  // Undo the permutation and the removeBroadcast
  if (maybePermutation.has_value()) {
    auto invPerm = maybePermutation.value().inverse();
    srcVals = invPerm.apply(srcVals);
  }
  srcVals = broadcastAs(srcVals, regL);
  auto structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(srcVals.size(), llvmElemTy));
  src = packLLElements(loc, typeConverter, srcVals, rewriter, structTy);
  return success();
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const ppu::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Type llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getSrc(), llvmElemTy, rewriter);
    Value smemBase = smemObj.getBase();
    Attribute srcLayout = memDescType.getEncoding();

    // Try to lower to ppu.ldmatrix.swzl for AIU load
    if (auto sharedEnc = dyn_cast<PPUAIUSharedEncodingAttr>(srcLayout)) {
      Attribute dstLayout = dstTy.getEncoding();
      if (isa<DotOperandEncodingAttr>(dstLayout) &&
          isa<PPUMmaEncodingAttr>(
              cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
        return lowerAIUSharedToPPUDotOperand(op, adaptor, getTypeConverter(),
                                             rewriter);
      } else {
        return lowerAIUSharedToDistributed(op, adaptor, getTypeConverter(),
                                           rewriter);
      }
    }

    // Try to lower to ppu.ldmatrix
    bool lowered = false;
    Value val;
    for (bool transpose : {false, true}) {
      lowered = lowerPPULdMatrix(op.getLoc(), dstTy, memDescType, transpose,
                                 val, smemBase, llvmElemTy, rewriter,
                                 targetInfo, getTypeConverter())
                    .succeeded();
      if (lowered) {
        rewriter.replaceOp(op, val);
        return success();
      }
    }

    // Try to lower to ldmatrix
    auto *typeConverter = getTypeConverter();
    llvm::SmallVector<Value> values;
    auto regLayout = toLinearLayout(dstTy);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    auto structTy = LLVM::LLVMStructType::getLiteral(
        op.getLoc().getContext(), SmallVector<Type>(values.size(), llvmElemTy));
    auto value =
        packLLElements(op.getLoc(), typeConverter, values, rewriter, structTy);
    rewriter.replaceOp(op, value);
    return success();
  }

  LogicalResult
  lowerAIUSharedToDistributed(triton::gpu::LocalLoadOp op,
                              triton::gpu::LocalLoadOpAdaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemTy = typeConverter->convertType(dstTy.getElementType());

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of loadSharedToDistributed");
    auto dstDistributedLayout = dstTy.getEncoding();
    auto srcSharedLayout =
        cast<triton::gpu::PPUAIUSharedEncodingAttr>(srcTy.getEncoding());
    auto srcElemTy = srcTy.getElementType();
    auto dstElemTy = dstTy.getElementType();
    LDBG("loadSharedToDistributed elemTy "
         << elemTy << " srcElemTy " << srcElemTy << " dstElemTy " << dstElemTy);

    auto inOrd = llvm::to_vector(srcSharedLayout.getOrder());
    auto outOrd = triton::gpu::getOrder(dstTy);
    unsigned outVec =
        inOrd == outOrd ? triton::gpu::getContigPerThread(dstTy)[outOrd[0]] : 1;

    unsigned inVec = 128 / elemTy.getIntOrFloatBitWidth();
    unsigned minVec = std::min(outVec, inVec);
    unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
    SmallVector<Value> offsetVals = {
        LLVM::PPU::getStrides(smemObj, srcTy, loc, rewriter).size(),
        b.i32_val(0)};

    DenseMap<unsigned, Value> sharedPtrs;
    sharedPtrs = LLVM::PPU::getAIUSwizzledSharedPtrs(
        loc, targetInfo, outVec, dstTy, srcTy, srcSharedLayout, elemTy, smemObj,
        rewriter, offsetVals);

    assert(outElems % minVec == 0 && "Unexpected number of elements");
    unsigned numVecs = outElems / minVec;
    auto wordTy = vec_ty(elemTy, minVec);
    SmallVector<Value> outVals(outElems);
    for (unsigned i = 0; i < numVecs; ++i) {
      Value smemAddr = sharedPtrs[i * minVec];
      smemAddr = b.bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
      auto valVec = b.load(wordTy, smemAddr);
      valVec.setAlignment(minVec * elemTy.getIntOrFloatBitWidth() / 8);
      for (unsigned v = 0; v < minVec; ++v) {
        Value currVal = b.extract_element(elemTy, valVec, b.i32_val(v));
        outVals[i * minVec + v] = currVal;
      }
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult
  lowerAIUSharedToPPUDotOperand(triton::gpu::LocalLoadOp op,
                                triton::gpu::LocalLoadOpAdaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto src = op.getSrc();
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dotEnc = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto mmaEncoding =
        dyn_cast<triton::gpu::PPUMmaEncodingAttr>(dotEnc.getParent());
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    Value res;
    if (auto sharedEnc = cast<PPUAIUSharedEncodingAttr>(srcTy.getEncoding())) {
      auto aiuLoad = sharedEnc.getAIUStrategy();
      if (mmaEncoding.isPPU0010()) {
        res = SharedToDotOperandPPUAIUV1::convertLayout(
            dotEnc.getOpIdx(), rewriter, loc, src, dotEnc, smemObj,
            getTypeConverter(), getThreadId(rewriter, loc), aiuLoad);
      } else if (mmaEncoding.isPPU0015()) {
        res = SharedToDotOperandPPUAIUV2::convertLayout(
            dotEnc.getOpIdx(), rewriter, loc, src, dotEnc, smemObj,
            getTypeConverter(), getThreadId(rewriter, loc), aiuLoad);
      } else {
        assert(false && "Unsupported mma layout found");
      }
    } else if (auto sharedEnc =
                   cast<SwizzledSharedEncodingAttr>(srcTy.getEncoding())) {
      // TODO old version for compute the address
      assert(false && "Unsupported mma layout found");
    }

    auto elemsI32 = unpackLLElements(loc, res, rewriter);
    // Unpack i32 values to the original type
    SmallVector<Value> elems;
    auto numElemsPerVec = 32 / bitwidth;
    auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
    for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
      auto vec = b.bitcast(elemsI32[v], vecTy);
      for (int i = 0; i < numElemsPerVec; ++i)
        elems.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
    }

    auto structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(elems.size(), llvmElemTy));
    auto ret = packLLElements(loc, typeConverter, elems, rewriter, structTy);
    rewriter.replaceOp(op, ret);
    return success();
  }

private:
  const ppu::TargetInfo &targetInfo;
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const ppu::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    MemDescType memDescType = op.getType();
    RankedTensorType regTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(regTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);
    auto smemObj = SharedMemoryObject(
        smemBase, llvmElemTy, memDescType.getRank(), op.getLoc(), rewriter);

    auto regLayout = toLinearLayout(regTy);
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }

    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const ppu::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const ppu::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType memDescType = op.getDst().getType();
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    auto regLayout = toLinearLayout(srcTy);
    auto values = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
    auto result =
        lowerLdStMatrix(op.getLoc(), regLayout, memDescType, values, smemObj,
                        rewriter, targetInfo, getTypeConverter());
    if (failed(result)) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const ppu::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::ppu::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Backend optimized memory ops get higher benefit
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
