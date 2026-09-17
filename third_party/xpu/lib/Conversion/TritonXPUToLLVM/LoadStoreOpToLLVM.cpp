#include "PatternTritonXPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const xpu::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {
    isBf16RoundToMid =
        mlir::triton::tools::getBoolEnvXPU("TRITONXPU_BF16_ROUND_MID");
    isBf16Fast = mlir::triton::tools::getBoolEnvXPU("TRITONXPU_BF16_FAST");
  }

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // The maximum vector size is 512 bits on XPUs.
    return std::min<unsigned>(512 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

  void getVectorInfo(Type tensorType, unsigned &vecSize,
                     unsigned &elemNbits) const {
    vecSize = 1u;
    elemNbits = 32u;
    if (auto vecType = mlir::dyn_cast<mlir::VectorType>(tensorType)) {
      unsigned numElems = vecType.getNumElements();
      Type elemTy = vecType.getElementType();
      elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                      ? 64u
                      : elemTy.getIntOrFloatBitWidth();
      // The maximum vector size is 512 bits on XPU2.
      vecSize = std::min<unsigned>(512 / elemNbits, numElems);
    }
  }

  void getLayoutInfo(Type type, size_t &ngroup, size_t &groupsize) const {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      if (auto globalEncoding = dyn_cast<triton::xpu::ClusterLayoutAttr>(
              tensorType.getEncoding())) {
        auto shape = tensorType.getShape();
        ngroup = product(globalEncoding.getGroupsPerCluster());
        groupsize = product(globalEncoding.getCoresPerGroup());
      }
    } else {
      ngroup = 1;
      groupsize = 1;
    }
  }

  Value bf16ToFp16(ConversionPatternRewriter &rewriter, mlir::Location &loc,
                   Type type, Value elem) const {
    Value convertElem = elem;
    if (auto vecType = mlir::dyn_cast<mlir::VectorType>(type)) {
      unsigned numElems = vecType.getNumElements();
      Type elemTy = vecType.getElementType();
      if (elemTy.isBF16()) {
        Type convertTy = VectorType::get(numElems, f16_ty);
        convertElem = bitcast(elem, convertTy);
      }
    }
    return convertElem;
  }

  void setHaddr(ConversionPatternRewriter &rewriter, mlir::Location &loc,
                Value ptr) const {
    Value ptrInt = ptrtoint(i64_ty, ptr);
    Value ptrIntH32 = lshr(ptrInt, int_val(64, 32));
    Value ptrIntS32 = trunc(i32_ty, ptrIntH32);
    rewriter.create<mlir::LLVM::XPU::SetHaddrOp>(loc, ptrIntS32);
  }

  void createGM2LMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, src);
      Value srcAs0 = addrspace_cast(ptr_ty(ctx, 0), src);
      rewriter.create<mlir::LLVM::XPU::GM2LMOp>(loc, srcAs0, dst, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::GM2LMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create GM2LMOp with unsupported xpu architecture.");
    }
  }

  void createLM2GMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, dst);
      Value dstAs0 = addrspace_cast(ptr_ty(ctx, 0), dst);
      rewriter.create<mlir::LLVM::XPU::LM2GMOp>(loc, src, dstAs0, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::LM2GMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create LM2GMOp with unsupported xpu architecture.");
    }
  }

  void createSM2GMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      setHaddr(rewriter, loc, dst);
      Value dstAs0 = addrspace_cast(ptr_ty(ctx, 0), dst);
      rewriter.create<mlir::LLVM::XPU::SM2GMOp>(loc, src, dstAs0, offset, size);
      break;
    }
    case XPUArch::XPU3: {
      rewriter.create<mlir::LLVM::XPU::SM2GMOp_v3>(loc, src, dst, offset, size);
      break;
    }
    default:
      llvm_unreachable(
          "Failed to create LM2GMOp with unsupported xpu architecture.");
    }
  }

  void createGM2SMOp(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value src,
                     Value dst, Value offset, Value size) const {
    rewriter.create<mlir::LLVM::XPU::GM2SMOp_v3>(loc, src, dst, offset, size);
  }

  void createMemOp(ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx,
                   mlir::Location &loc, Value bufPtr, Value gmPtr, Value offset,
                   Value size, MemCpyType memCpyType) const {
    switch (static_cast<MemCpyType>(memCpyType)) {
    case MemCpyType::GM2LM:
      createGM2LMOp(rewriter, ctx, loc, bufPtr, gmPtr, offset, size);
      break;
    case MemCpyType::LM2GM:
      createLM2GMOp(rewriter, ctx, loc, gmPtr, bufPtr, offset, size);
      break;
    case MemCpyType::SM2GM:
      createSM2GMOp(rewriter, ctx, loc, bufPtr, gmPtr, offset, size);
      break;
    default:
      llvm_unreachable("Memory Op only includes GM2LM, LM2GM, SM2GM");
    }
  }

  void createMfenceOp(ConversionPatternRewriter &rewriter, mlir::Location &loc,
                      int32_t mfenceType = 5) const {
    // Mfence mask bits: bit0=LM(1), bit1=SM(2), bit2=GM(4). 5 fences LM+GM.
    rewriter.create<mlir::LLVM::XPU::MfenceOp>(loc, i32_val(mfenceType));
  }

  void createMfenceLMOp(ConversionPatternRewriter &rewriter,
                        mlir::Location &loc) const {
    createMfenceOp(rewriter, loc, 1);
  }

  void createMfenceGMOp(ConversionPatternRewriter &rewriter,
                        mlir::Location &loc) const {
    createMfenceOp(rewriter, loc, 4);
  }

  Value getStartPtr(ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx,
                    mlir::Location &loc, Value gmPtr, Value zeroPtr,
                    Value rowLen, Value elemBytes) const {
    Value gmPtrInt = ptrtoint(i64_ty, gmPtr);
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value offset = sdiv(sub(gmPtrInt, zeroPtrInt), elemBytes);
    Value startOffsetBytes = mul(mul(sdiv(offset, rowLen), rowLen), elemBytes);
    Value startPtr = gep(ptr_ty(ctx, 0), i8_ty, zeroPtr,
                         startOffsetBytes); // convert ptr first, then move
    return startPtr;
  }

  Value getReadBytes(ConversionPatternRewriter &rewriter,
                     mlir::MLIRContext *ctx, mlir::Location &loc, Value readLen,
                     Value llMask, Value llLen, Value mask, Value len,
                     Value elemBytes) const {
    Value _readLen =
        readLen.getType().isInteger(64) ? trunc(i32_ty, readLen) : readLen;
    Value _len = len;
    if (llLen) {
      _len = _len.getType().isInteger(64) ? trunc(i32_ty, _len) : _len;
    }
    Value _elemBytes = elemBytes.getType().isInteger(64)
                           ? trunc(i32_ty, elemBytes)
                           : elemBytes;
    _readLen = llLen ? smin(smax(_len, i32_val(0)), _readLen) : _readLen;
    Value readBytes = mul(_readLen, _elemBytes);
    readBytes = llMask ? select(mask, readBytes, i32_val(0)) : readBytes;
    return readBytes;
  }

  /* ******************************** without mask zero
   * *****************************************/
  void lowerLocallyContinuousUnfixedStride(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      int64_t _rowLen, int64_t _bufLen, int64_t _elemBytes, Value llGMPtr,
      Value llLMPtr, Value llLen, Value offsetBytes, MemCpyType memCpyType,
      Block *oldBlock, Block *newBlock) const {
    // clang-format off
    /* *****************************************************************************
    def getStartPtr(gmPtr, zeroPtr, rowLen, elemBytes):
        offset = (gmPtr - zeroPtr) / elemBytes
        startOffsetBytes = (offset / rowLen) * rowLen * elemBytes
        return zeroPtr + startOffsetBytes

    _rowMaxTail = _bufLen % _rowLen
    _rowNum = _bufLen / _rowLen
    rowBytes = rowLen * elemBytes
    tailLen = min(rowLen - (gmPtr.front() - zeroPtr) / elemBytes % rowLen, bufLen)
    if _rowMaxTail == 0:
      for i in range(_rowNum):
        gmStartPtr = llGMPtrs[i * _rowLen]
        lmOffsetBytes = (i * _rowLen) * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes;
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    else:
      if 0 < tailLen < rowMaxTail:
        gm2lm(gmPtr.front(), lmPtr, tailBytes)
        for i in range(_rowNum):
          gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+i*_rowLen], zeroPtr, rowLen, elemBytes)
          lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
          lmStartPtr = lmPtr + lmOffsetBytes
          gm2lm(gmStartPtr, lmStartPtr, rowBytes)
        gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
        offset = tailLen + rowNum * rowLen
        lmOffsetBytes = offset * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes
        remainBytes = (bufLen - offset) * elemBytes
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
      else:
          gm2lm(gmPtr.front(), lmPtr, tailBytes)
          if _rowNum >= 1:
            for i in range(_rowNum-1):
              gmPtr1 = gmPtr[_rowMaxTail+i*_rowLen]
              gmPtr2 = gmPtr[_rowMaxTail+(i+1)*_rowLen]
              gmPtr = select(tailLen == rowMaxTail, gmPtr1, gmPtr2)
              gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+(i+1)*_rowLen], zeroPtr, rowLen, elemBytes)
              lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
              lmStartPtr = lmPtr + lmOffsetBytes
              gm2lm(gmStartPtr, lmStartPtr, rowBytes)
            gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
            offset = tailLen + (rowNum - 1) * rowLen
            lmOffsetBytes = offset * elemBytes
            lmStartPtr = lmPtr + lmOffsetBytes
            remainBytes = (bufLen - offset) * elemBytes
            gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    ********************************************************************************/
    // clang-format on
    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    Value gmFrontPtr = llGMPtrs.front();
    Value gmBackPtr = llGMPtrs.back();
    Value lmPtr = llLMPtrs.front();

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(gmFrontPtr);
    Value zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value gmFrontPtrInt = ptrtoint(i64_ty, gmFrontPtr);

    int64_t _rowMaxTail = _bufLen % _rowLen;
    int64_t _rowNum = _bufLen / _rowLen;
    Value rowMaxTail = i64_val(_rowMaxTail);
    Value rowNum = i64_val(_rowNum);
    Value rowLen = i64_val(_rowLen);
    Value bufLen = i64_val(_bufLen);
    Value elemBytes = i64_val(_elemBytes);
    Value rowBytes = trunc(i32_ty, mul(rowLen, elemBytes));

    if (_rowMaxTail == 0) {
      // GM2LM/LM2GM Row Data
      for (int64_t i = 0; i < _rowNum; ++i) {
        Value gmStartPtr = llGMPtrs[i * _rowLen];
        Value lmOffsetBytes = mul(i64_val(i * _rowLen), elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    rowBytes, memCpyType);
      }
    } else {
      Value gmFrontOffset = sdiv(sub(gmFrontPtrInt, zeroPtrInt), elemBytes);
      Value tailLen = smin(sub(rowLen, srem(gmFrontOffset, rowLen)), bufLen);

      Block *thenBB = rewriter.createBlock(newBlock);
      Block *elseBB = rewriter.createBlock(newBlock);
      Block *mfenceBB = rewriter.createBlock(newBlock);
      rewriter.setInsertionPointToEnd(oldBlock);

      Value condTailSgt = icmp_sgt(tailLen, i64_val(0));
      Value condTailSlt = icmp_slt(tailLen, rowMaxTail);
      Value condTailDiff = and_(condTailSgt, condTailSlt);
      rewriter.create<LLVM::CondBrOp>(loc, condTailDiff, thenBB, elseBB);
      // 1. ThenBB
      rewriter.setInsertionPointToEnd(thenBB);
      {
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailBytes = trunc(i32_ty, mul(tailLen, elemBytes));
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    tailBytes, memCpyType);
        // 1.2 GM2LM/LM2GM Row Data
        for (int64_t i = 0; i < _rowNum; ++i) {
          Value gmPtr = llGMPtrs[_rowMaxTail + i * _rowLen];
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value lmOffsetBytes =
              mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      rowBytes, memCpyType);
        }
        // 1.3 GM2LM/LM2GM Remain Data
        Value gmPtr = llGMPtrs.back();
        Value gmStartPtr =
            getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr, rowLen, elemBytes);
        Value offset = add(tailLen, i64_val(_rowNum * _rowLen));
        Value lmOffsetBytes = mul(offset, elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        Value remainBytes = trunc(i32_ty, mul(sub(bufLen, offset), elemBytes));
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    remainBytes, memCpyType);
      }
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB

      // 2. elseBB
      rewriter.setInsertionPointToEnd(elseBB);
      {
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailBytes = trunc(i32_ty, mul(tailLen, elemBytes));
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    tailBytes, memCpyType);
        if (_rowNum >= 1) {
          // 1.2 GM2LM/LM2GM Row Data
          Value gmCond = icmp_eq(tailLen, rowMaxTail);
          for (int64_t i = 0; i < _rowNum - 1; ++i) {
            Value gmPtr1 = llGMPtrs[_rowMaxTail + i * _rowLen];
            Value gmPtr2 = llGMPtrs[_rowMaxTail + (i + 1) * _rowLen];
            Value gmPtr = select(gmCond, gmPtr1, gmPtr2);
            Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                           rowLen, elemBytes);
            Value lmOffsetBytes =
                mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
            Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
            createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                        rowBytes, memCpyType);
          }
          // 1.3 GM2LM/LM2GM Remain Data
          Value gmPtr = llGMPtrs.back();
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value offset = add(tailLen, i64_val((_rowNum - 1) * _rowLen));
          Value lmOffsetBytes = mul(offset, elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          Value remainBytes =
              trunc(i32_ty, mul(sub(bufLen, offset), elemBytes));
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      remainBytes, memCpyType);
        }
      }
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB

      // 3. mefenceBB
      rewriter.setInsertionPointToEnd(mfenceBB);
    }
  }

  void lowerLocallyContinuousLargeRow(Operation *op, Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      size_t rowSize, size_t rowStride,
                                      Value llGMPtr, Value llLMPtr, Value llLen,
                                      Value bufLen, Value elemBytes,
                                      Value offsetBytes, MemCpyType memCpyType,
                                      Block *oldBlock, Block *newBlock) const {

    /* *************************************************
    gapLen = strideLen - rowLen
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen) * rowLen
    realTailLen = rowLen - (bankOffset - (blockOffset + rowOffset))

    if 0 < realTailLen < bufLen:
      gm2lm(bankPtr, lmPtr, realTailLen * elemBytes)
      gm2lm(bankPtr + (realTailLen + gapLen) * elemBytes, lmPtr + realTailLen,
    elemBytes,（bufLen - realTailLen）* elemBytes)

    else :
      gm2lm(bankPtr, lmPtr, bufLen * elemBytes)
    * ************************************************/

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    size_t gapSize = rowStride - rowSize;
    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value gapLen = i32_val(gapSize);
    Value gapBytes = mul(gapLen, elemBytes);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value realTailLen =
        sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(realTailLen, elemBytes);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *thenBB = rewriter.createBlock(newBlock);
    Block *elseBB = rewriter.createBlock(newBlock);
    Block *mfenceBB = rewriter.createBlock(newBlock);
    rewriter.setInsertionPointToEnd(oldBlock);

    Value condRemSgt = icmp_sgt(realTailLen, i32_val(0));
    Value condRemSlt = icmp_slt(realTailLen, bufLen);
    Value condRemDiff = and_(condRemSgt, condRemSlt);
    rewriter.create<LLVM::CondBrOp>(loc, condRemDiff, thenBB, elseBB);
    rewriter.setInsertionPointToEnd(thenBB);
    // 1. ThenBB
    // 1.1 GM2LM Tail Data
    Value tailLen = realTailLen;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        Value limitedLen =
            smin(smax(llLens[0], i64_val(0)), sext(i64_ty, bufLen));
        tailLen = smin(realTailLen, trunc(i32_ty, limitedLen));
      } else if (llLens[0].getType().isInteger(1)) {
        Value limitedLen = bufLen;
        tailLen = smin(realTailLen, limitedLen);
      } else {
        Value limitedLen = smin(smax(llLens[0], i32_val(0)), bufLen);
        tailLen = smin(realTailLen, limitedLen);
      }
    }
    Value tailBytes = mul(tailLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, tailBytes,
                memCpyType);

    // 1.2 GM2LM Remain Data
    Value startCond;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        startCond = icmp_sge(sext(i64_ty, realTailLen), llLens[0]);
      } else if (llLens[0].getType().isInteger(1)) {
        startCond = icmp_sge(realTailLen, bufLen);
      } else {
        startCond = icmp_sge(realTailLen, llLens[0]);
      }
    }
    Value startPtrInt =
        add(bankPtrInt, zext(i64_ty, add(realTailBytes, gapBytes)));
    Value startPtr =
        rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
    startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
    Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                            realTailBytes); // convert ptr first, then move

    Value remainLen = sub(bufLen, realTailLen);
    Value remainBytes = mul(remainLen, elemBytes);
    createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                remainBytes, memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                mfenceBB); // Jump to mfenceBB

    // 2. elseBB
    rewriter.setInsertionPointToEnd(elseBB);
    // GM2LM the whole bufLen
    Value readBytes = mul(bufLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                mfenceBB); // Jump to mfenceBB

    // 3. mefenceBB
    rewriter.setInsertionPointToEnd(mfenceBB);
  }

  void lowerLocallyContinuousSmallRow(Operation *op, Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      size_t rowSize, size_t rowStride,
                                      Value llGMPtr, Value llLMPtr, Value llLen,
                                      Value bufLen, Value elemBytes,
                                      Value offsetBytes, MemCpyType memCpyType,
                                      Block *oldBlock, Block *newBlock) const {

    /* *************************************************
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen)
    rowHeadLen = bankOffset - (blockOffset + rowOffset)
    realTailLen = rowLen - rowHeadLen
    rowNum = (bufLen - realTailLen - 1) / rowLen

    gm2lm(bankPtr, lmPtr, realTailLen * elemBytes)

    for(i = 0; i < rowNum; i++) {
        gm2lm(bankPtr + ((i + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (realTailLen + i * rowLen) * elemBytes, rowLen * elemBytes)
    }

    remLen = bufLen - realTailLen - rowNum * rowLen
    gm2lm(bankPtr + ((rowNum + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (realTailLen + rowNum * rowLen) * elemBytes, (remLen * elemBytes)
    *************************************************/

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }
    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value realTailLen =
        sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(realTailLen, elemBytes);
    Value rowBytes = mul(rowLen, elemBytes);
    Value rowHeadLen = sub(rowLen, realTailLen);
    Value rowHeadBytes = sub(rowBytes, realTailBytes);
    Value realRemainLen = sub(sub(bufLen, realTailLen), i32_val(1));
    Value rowNum = sdiv(realRemainLen, rowLen);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *judgeBB = rewriter.createBlock(newBlock, TypeRange{i32_ty}, {loc});
    Block *gm2lmRowBB = rewriter.createBlock(newBlock);
    Block *stepBB = rewriter.createBlock(newBlock);
    Block *gm2lmRemBB = rewriter.createBlock(newBlock);

    // 1.  GM2LM Tail Data
    rewriter.setInsertionPointToEnd(oldBlock);
    Value tailLen = realTailLen;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        tailLen = smin(realTailLen, trunc(i32_ty, llLens[0]));
      } else {
        tailLen = smin(realTailLen, llLens[0]);
      }
    }
    Value tailBytes = mul(tailLen, elemBytes);
    createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, tailBytes,
                memCpyType);

    Value _init = i32_val(0);
    Value _step = i32_val(1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{_init},
                                judgeBB); // Jump to judgeBB
    Value iter = judgeBB->getArgument(0);

    // 2. GM2LM Row Data
    rewriter.setInsertionPointToEnd(judgeBB);
    Value condSlt = icmp_slt(iter, rowNum);
    rewriter.create<LLVM::CondBrOp>(loc, condSlt, gm2lmRowBB, gm2lmRemBB);

    rewriter.setInsertionPointToEnd(gm2lmRowBB);
    Value skipStride = mul(add(iter, i32_val(1)), strideLen);
    Value skipStrideBytes = mul(skipStride, elemBytes);
    Value skipRowLen = mul(iter, rowLen);
    Value startPtrInt =
        add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
    Value startPtr =
        rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
    startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
    Value startCond;
    if (llLen) {
      auto llLens = unpackLLElements(loc, llLen, rewriter);
      if (llLens[0].getType().isInteger(64)) {
        startCond =
            icmp_sgt(sext(i64_ty, add(skipRowLen, realTailLen)), llLens[0]);
      } else {
        startCond = icmp_sgt(add(skipRowLen, realTailLen), llLens[0]);
      }
    }
    startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
    Value dstOffset = add(realTailLen, skipRowLen);
    Value dstOffsetBytes = mul(dstOffset, elemBytes);
    Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                            dstOffsetBytes); // convert ptr first, then move
    createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                rowBytes, memCpyType);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, stepBB); // Jump to stepBB

    rewriter.setInsertionPointToEnd(stepBB);
    Value _index = add(iter, _step);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{_index},
                                judgeBB); // Jump back to judgeBB

    // 3 GM2LM Remain Data
    rewriter.setInsertionPointToEnd(gm2lmRemBB);
    {
      Value skipStride = mul(add(rowNum, i32_val(1)), strideLen);
      Value skipStrideBytes = mul(skipStride, elemBytes);
      Value skipRowLen = mul(rowNum, rowLen);
      Value remainBytes =
          mul(sub(bufLen, add(realTailLen, skipRowLen)), elemBytes);
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
      Value startCond;
      if (llLen) {
        auto llLens = unpackLLElements(loc, llLen, rewriter);
        if (llLens[0].getType().isInteger(64)) {
          startCond =
              icmp_sgt(sext(i64_ty, add(skipRowLen, realTailLen)), llLens[0]);
        } else {
          startCond = icmp_sgt(add(skipRowLen, realTailLen), llLens[0]);
        }
      }
      startPtr = startCond ? select(startCond, zeroPtr, startPtr) : startPtr;
      Value dstOffset = add(realTailLen, skipRowLen);
      Value dstOffsetBytes = mul(dstOffset, elemBytes);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              dstOffsetBytes); // convert ptr first, then move
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  remainBytes, memCpyType);
    }
  }

  /* ******************************** with mask zero
   * *****************************************/
  void lowerLocallyContinuousUnfixedStrideMask(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      int64_t _rowLen, int64_t _bufLen, int64_t _elemBytes, Value llGMPtr,
      Value llLMPtr, Value llMask, Value llLen, Value offsetBytes,
      MemCpyType memCpyType, Block *oldBlock, Block *newBlock) const {
    // clang-format off
    /* *****************************************************************************
    def getStartPtr(gmPtr, zeroPtr, rowLen, elemBytes):
        offset = (gmPtr - zeroPtr) / elemBytes
        startOffsetBytes = (offset / rowLen) * rowLen * elemBytes
        return zeroPtr + startOffsetBytes

    _rowMaxTail = _bufLen % _rowLen
    _rowNum = _bufLen / _rowLen
    rowBytes = rowLen * elemBytes
    tailLen = min(rowLen - (gmPtr.front() - zeroPtr) / elemBytes % rowLen, bufLen)
    if _rowMaxTail == 0:
      for i in range(_rowNum):
        gmStartPtr = llGMPtrs[i * _rowLen]
        lmOffsetBytes = (i * _rowLen) * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes;
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    else:
      if 0 < tailLen < rowMaxTail:
        gm2lm(gmPtr.front(), lmPtr, tailBytes)
        for i in range(_rowNum):
          gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+i*_rowLen], zeroPtr, rowLen, elemBytes)
          lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
          lmStartPtr = lmPtr + lmOffsetBytes
          gm2lm(gmStartPtr, lmStartPtr, rowBytes)
        gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
        offset = tailLen + rowNum * rowLen
        lmOffsetBytes = offset * elemBytes
        lmStartPtr = lmPtr + lmOffsetBytes
        remainBytes = (bufLen - offset) * elemBytes
        gm2lm(gmStartPtr, lmStartPtr, remainBytes)
      else:
          gm2lm(gmPtr.front(), lmPtr, tailBytes)
          if _rowNum >= 1:
            for i in range(_rowNum-1):
              gmPtr1 = gmPtr[_rowMaxTail+i*_rowLen]
              gmPtr2 = gmPtr[_rowMaxTail+(i+1)*_rowLen]
              gmPtr = select(tailLen == rowMaxTail, gmPtr1, gmPtr2)
              gmStartPtr = getStartPtr(gmPtr[_rowMaxTail+(i+1)*_rowLen], zeroPtr, rowLen, elemBytes)
              lmOffsetBytes = (tailLen + i * rowLen) * elemBytes
              lmStartPtr = lmPtr + lmOffsetBytes
              gm2lm(gmStartPtr, lmStartPtr, rowBytes)
            gmStartPtr = getStartPtr(gmPtr.back(), zeroPtr, rowLen, elemBytes)
            offset = tailLen + (rowNum - 1) * rowLen
            lmOffsetBytes = offset * elemBytes
            lmStartPtr = lmPtr + lmOffsetBytes
            remainBytes = (bufLen - offset) * elemBytes
            gm2lm(gmStartPtr, lmStartPtr, remainBytes)
    ********************************************************************************/
    // clang-format on
    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    SmallVector<Value> llMasks;
    if (llMask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }
    SmallVector<Value> llLens;
    if (llLen) {
      llLens = unpackLLElements(loc, llLen, rewriter);
    }

    Value gmFrontPtr = llGMPtrs.front();
    Value gmBackPtr = llGMPtrs.back();
    Value lmPtr = llLMPtrs.front();

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(gmFrontPtr);
    Value zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value gmFrontPtrInt = ptrtoint(i64_ty, gmFrontPtr);

    int64_t _rowNum = _bufLen / _rowLen;
    int64_t _rowMaxTail = _rowNum > 0 ? _bufLen % _rowLen : _bufLen - 1;
    Value rowMaxTail = i64_val(_rowMaxTail);
    Value rowNum = i64_val(_rowNum);
    Value rowLen = i64_val(_rowLen);
    Value bufLen = i64_val(_bufLen);
    Value elemBytes = i64_val(_elemBytes);

    if (_rowMaxTail == 0) {
      // GM2LM/LM2GM Row Data
      for (int64_t i = 0; i < _rowNum; ++i) {
        Value gmStartPtr = llGMPtrs[i * _rowLen];
        Value lmOffsetBytes = mul(i64_val(i * _rowLen), elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        Value mask = llMask ? llMasks[i * _rowLen] : Value();
        Value len = llLen ? llLens[i * _rowLen] : Value();
        Value readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask,
                                       llLen, mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    readBytes, memCpyType);
      }
    } else {
      Value gmFrontOffset = sdiv(sub(gmFrontPtrInt, zeroPtrInt), elemBytes);
      Value tailLen = smin(sub(rowLen, srem(gmFrontOffset, rowLen)), bufLen);

      Block *thenBB = rewriter.createBlock(newBlock);
      Block *elseBB = rewriter.createBlock(newBlock);
      Block *mfenceBB = rewriter.createBlock(newBlock);
      rewriter.setInsertionPointToEnd(oldBlock);

      Value condTailSgt = icmp_sgt(tailLen, i64_val(0));
      Value condTailSlt = icmp_slt(tailLen, rowMaxTail);
      Value condTailDiff = and_(condTailSgt, condTailSlt);
      rewriter.create<LLVM::CondBrOp>(loc, condTailDiff, thenBB, elseBB);
      // 1. ThenBB
      {
        rewriter.setInsertionPointToEnd(thenBB);
        // 1.1 GM2LM/LM2GM Tail Data
        Value mask = llMask ? llMasks[0] : Value();
        Value len = llLen ? llLens[0] : Value();
        Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask,
                                       llLen, mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    readBytes, memCpyType);
        // 1.2 GM2LM/LM2GM Row Data
        for (int64_t i = 0; i < _rowNum; ++i) {
          Value gmPtr = llGMPtrs[_rowMaxTail + i * _rowLen];
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value lmOffsetBytes =
              mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          mask = llMask ? llMasks[_rowMaxTail + i * _rowLen] : Value();
          len = llLen ? llLens[_rowMaxTail + i * _rowLen] : Value();
          readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask, llLen,
                                   mask, len, elemBytes);
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      readBytes, memCpyType);
        }
        // 1.3 GM2LM/LM2GM Remain Data
        Value gmPtr = llGMPtrs.back();
        Value gmStartPtr =
            getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr, rowLen, elemBytes);
        Value offset = add(tailLen, i64_val(_rowNum * _rowLen));
        Value lmOffsetBytes = mul(offset, elemBytes);
        Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
        Value remainLen = sub(bufLen, offset);
        mask = llMask ? llMasks[_rowMaxTail + _rowNum * _rowLen] : Value();
        len = llLen ? llLens[_rowMaxTail + _rowNum * _rowLen] : Value();
        readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask, llLen,
                                 mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                    readBytes, memCpyType);

        rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                    mfenceBB); // Jump to mfenceBB
      }

      // 2. elseBB
      {
        rewriter.setInsertionPointToEnd(elseBB);
        // 1.1 GM2LM/LM2GM Tail Data
        Value tailLen = mul(tailLen, elemBytes);
        Value mask = llMask ? llMasks[0] : Value();
        Value len = llLen ? llLens[0] : Value();
        Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask,
                                       llLen, mask, len, elemBytes);
        createMemOp(rewriter, ctx, loc, gmFrontPtr, lmPtr, offsetBytes,
                    readBytes, memCpyType);
        if (_rowNum >= 1) {
          // 1.2 GM2LM/LM2GM Row Data
          Value gmCond = icmp_eq(tailLen, rowMaxTail);
          for (int64_t i = 0; i < _rowNum - 1; ++i) {
            Value gmPtr1 = llGMPtrs[_rowMaxTail + i * _rowLen];
            Value gmPtr2 = llGMPtrs[_rowMaxTail + (i + 1) * _rowLen];
            Value gmPtr = select(gmCond, gmPtr1, gmPtr2);
            Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                           rowLen, elemBytes);
            Value lmOffsetBytes =
                mul(add(tailLen, i64_val(i * _rowLen)), elemBytes);
            Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);

            Value mask1 = llMask ? llMasks[_rowMaxTail + i * _rowLen] : Value();
            Value mask2 =
                llMask ? llMasks[_rowMaxTail + (i + 1) * _rowLen] : Value();
            mask = llMask ? select(gmCond, mask1, mask2) : Value();

            Value len1 = llLen ? llLens[_rowMaxTail + i * _rowLen] : Value();
            Value len2 =
                llLen ? llLens[_rowMaxTail + (i + 1) * _rowLen] : Value();
            len = llLen ? select(gmCond, len1, len2) : Value();
            readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask, llLen,
                                     mask, len, elemBytes);
            createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                        readBytes, memCpyType);
          }
          // 1.3 GM2LM/LM2GM Remain Data
          Value gmPtr = llGMPtrs.back();
          Value gmStartPtr = getStartPtr(rewriter, ctx, loc, gmPtr, zeroPtr,
                                         rowLen, elemBytes);
          Value offset = add(tailLen, i64_val((_rowNum - 1) * _rowLen));
          Value lmOffsetBytes = mul(offset, elemBytes);
          Value lmStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr, lmOffsetBytes);
          Value remainLen = sub(bufLen, offset);
          mask = llMask ? llMasks[(_rowNum - 1) * _rowLen] : Value();
          len = llLen ? llLens[(_rowNum - 1) * _rowLen] : Value();
          readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask, llLen,
                                   mask, len, elemBytes);
          createMemOp(rewriter, ctx, loc, gmStartPtr, lmStartPtr, offsetBytes,
                      readBytes, memCpyType);
        }

        rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                    mfenceBB); // Jump to mfenceBB
      }

      // 3. mefenceBB
      rewriter.setInsertionPointToEnd(mfenceBB);
    }
  }

  void lowerLocallyContinuousUnfixedStrideMask(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      size_t rowSize, size_t rowStride, Value llGMPtr, Value llLMPtr,
      Value llMask, Value llLen, Value bufLen, Value elemBytes,
      Value offsetBytes, MemCpyType memCpyType, Block *oldBlock,
      Block *newBlock) const {
    // clang-format off
    /* *************************************************
    gapLen = strideLen - rowLen
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen) * rowLen
    tailLen = rowLen - (bankOffset - (blockOffset + rowOffset))

    if 0 < tailLen < bufLen:
      gm2lm(bankPtr, lmPtr, tailLen * elemBytes)
      gm2lm(bankPtr + (tailLen + gapLen) * elemBytes, lmPtr + tailLen,
    elemBytes,（bufLen - tailLen）* elemBytes)

    else :
      gm2lm(bankPtr, lmPtr, bufLen * elemBytes)
    * ************************************************/
    // clang-format on

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    SmallVector<Value> llMasks;
    if (llMask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }
    SmallVector<Value> llLens;
    if (llLen) {
      llLens = unpackLLElements(loc, llLen, rewriter);
    }

    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }

    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    size_t gapSize = rowStride - rowSize;
    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value gapLen = i32_val(gapSize);
    Value gapBytes = mul(gapLen, elemBytes);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value tailLen = sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value tailBytes = mul(tailLen, elemBytes);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *thenBB = rewriter.createBlock(newBlock);
    Block *elseBB = rewriter.createBlock(newBlock);
    Block *mfenceBB = rewriter.createBlock(newBlock);
    rewriter.setInsertionPointToEnd(oldBlock);

    Value condRemSgt = icmp_sgt(tailLen, i32_val(0));
    Value condRemSlt = icmp_slt(tailLen, bufLen);
    Value condRemDiff = and_(condRemSgt, condRemSlt);
    rewriter.create<LLVM::CondBrOp>(loc, condRemDiff, thenBB, elseBB);
    rewriter.setInsertionPointToEnd(thenBB);
    // 1. ThenBB
    {
      // 1.1 GM2LM Tail Data
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                  memCpyType);

      // 1.2 GM2LM Remain Data
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, add(tailBytes, gapBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              tailBytes); // convert ptr first, then move

      Value remainLen = sub(bufLen, tailLen);
      Value remainBytes = mul(remainLen, elemBytes);
      mask = llMask ? llMasks.back() : Value();
      len = llLen ? llLens.back() : Value();
      readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask, llLen,
                               mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  readBytes, memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB
    }

    // 2. elseBB
    {
      rewriter.setInsertionPointToEnd(elseBB);
      // GM2LM the whole bufLen
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, bufLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                  memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{},
                                  mfenceBB); // Jump to mfenceBB
    }

    // 3. mefenceBB
    rewriter.setInsertionPointToEnd(mfenceBB);
  }

  void lowerLocallyContinuousSmallRowMask(
      Operation *op, Location loc, ConversionPatternRewriter &rewriter,
      size_t rowSize, size_t rowStride, Value llGMPtr, Value llLMPtr,
      Value llMask, Value llLen, Value bufLen, Value elemBytes,
      Value offsetBytes, MemCpyType memCpyType, Block *oldBlock,
      Block *newBlock) const {
    // clang-format off
    /* *************************************************
    bankOffset = (bankPtrInt - zeroPtrInt) / elemBytes
    rowOffset = bankOffset / strideLen * strideLen
    blockOffset = ((bankOffset - rowOffset) / rowLen)
    rowHeadLen = bankOffset - (blockOffset + rowOffset)
    tailLen = rowLen - rowHeadLen
    rowNum = (bufLen - tailLen - 1) / rowLen

    gm2lm(bankPtr, lmPtr, tailLen * elemBytes)

    for(i = 0; i < rowNum; i++) {
        gm2lm(bankPtr + ((i + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (tailLen + i * rowLen) * elemBytes, rowLen * elemBytes)
    }

    remLen = bufLen - tailLen - rowNum * rowLen
    gm2lm(bankPtr + ((rowNum + 1) * strideLen - rowHeadLen) * elemBytes, lmPtr +
    (tailLen + rowNum * rowLen) * elemBytes, (remLen * elemBytes)
    *************************************************/
    // clang-format on

    MLIRContext *ctx = rewriter.getContext();

    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    SmallVector<Value> llMasks;
    if (llMask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }
    SmallVector<Value> llLens;
    if (llLen) {
      llLens = unpackLLElements(loc, llLen, rewriter);
    }

    auto bankPtr = llGMPtrs[0];
    auto lmBuf = llLMPtrs[0];
    if (bufLen.getType().isInteger(64)) {
      bufLen = trunc(i32_ty, bufLen);
    }
    auto zeroOp = findDefOpBwd<LLVM::GEPOp>(bankPtr);
    auto zeroPtr = cast<LLVM::GEPOp>(zeroOp).getBase();
    Value zeroPtrInt = ptrtoint(i64_ty, zeroPtr);
    Value bankPtrInt = ptrtoint(i64_ty, bankPtr);

    Value rowLen = i32_val(rowSize);
    Value strideLen = i32_val(rowStride);
    Value bankOffset =
        sdiv(trunc(i32_ty, sub(bankPtrInt, zeroPtrInt)), elemBytes);
    Value rowOffset = rowStride == 0
                          ? i32_val(0)
                          : mul(sdiv(bankOffset, strideLen), strideLen);
    Value blockOffset =
        rowStride == 0 ? i32_val(0)
                       : mul(sdiv(sub(bankOffset, rowOffset), rowLen), rowLen);
    Value tailLen = sub(rowLen, sub(bankOffset, add(blockOffset, rowOffset)));
    Value realTailBytes = mul(tailLen, elemBytes);
    Value rowBytes = mul(rowLen, elemBytes);
    Value rowHeadLen = sub(rowLen, tailLen);
    Value rowHeadBytes = sub(rowBytes, realTailBytes);
    Value realRemainLen = sub(sub(bufLen, tailLen), i32_val(1));
    Value rowNum = sdiv(realRemainLen, rowLen);

    zeroPtr = gep(ptr_ty(ctx, 1), i8_ty, zeroPtr, i32_val(0));
    bankPtr = gep(ptr_ty(ctx, 1), i8_ty, bankPtr, i32_val(0));
    Value lmPtr = gep(ptr_ty(ctx, 0), i8_ty, lmBuf, i32_val(0));

    Block *judgeBB = rewriter.createBlock(newBlock, TypeRange{i32_ty}, {loc});
    Block *gm2lmRowBB = rewriter.createBlock(newBlock);
    Block *stepBB = rewriter.createBlock(newBlock);
    Block *gm2lmRemBB = rewriter.createBlock(newBlock);

    // 1.  GM2LM Tail Data
    {
      rewriter.setInsertionPointToEnd(oldBlock);
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, tailLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, bankPtr, lmPtr, offsetBytes, readBytes,
                  memCpyType);
    }

    // 2. GM2LM Row Data
    {
      Value _init = i32_val(0);
      Value _step = i32_val(1);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{_init},
                                  judgeBB); // Jump to judgeBB
      Value iter = judgeBB->getArgument(0);

      rewriter.setInsertionPointToEnd(judgeBB);
      Value condSlt = icmp_slt(iter, rowNum);
      rewriter.create<LLVM::CondBrOp>(loc, condSlt, gm2lmRowBB, gm2lmRemBB);

      rewriter.setInsertionPointToEnd(gm2lmRowBB);
      Value skipStride = mul(add(iter, i32_val(1)), strideLen);
      Value skipStrideBytes = mul(skipStride, elemBytes);
      Value skipRowLen = mul(iter, rowLen);
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
      Value dstOffset = add(tailLen, skipRowLen);
      Value dstOffsetBytes = mul(dstOffset, elemBytes);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              dstOffsetBytes); // convert ptr first, then move
      Value mask = llMask ? llMasks[0] : Value();
      Value len = llLen ? llLens[0] : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, rowLen, llMask, llLen,
                                     mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  readBytes, memCpyType);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, stepBB); // Jump to stepBB

      rewriter.setInsertionPointToEnd(stepBB);
      Value _index = add(iter, _step);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{_index},
                                  judgeBB); // Jump back to judgeBB
    }

    // 3 GM2LM Remain Data
    {
      rewriter.setInsertionPointToEnd(gm2lmRemBB);

      Value skipStride = mul(add(rowNum, i32_val(1)), strideLen);
      Value skipStrideBytes = mul(skipStride, elemBytes);
      Value skipRowLen = mul(rowNum, rowLen);
      Value remainLen = sub(bufLen, add(tailLen, skipRowLen));
      Value startPtrInt =
          add(bankPtrInt, zext(i64_ty, sub(skipStrideBytes, rowHeadBytes)));
      Value startPtr =
          rowStride == 0 ? zeroPtr : inttoptr(ptr_ty(ctx, 1), startPtrInt);
      startPtr = gep(ptr_ty(ctx, 1), i8_ty, startPtr, i32_val(0));
      Value dstOffset = add(tailLen, skipRowLen);
      Value dstOffsetBytes = mul(dstOffset, elemBytes);
      Value dstStartPtr = gep(ptr_ty(ctx, 0), i8_ty, lmPtr,
                              dstOffsetBytes); // convert ptr first, then move
      Value mask = llMask ? llMasks.back() : Value();
      Value len = llLen ? llLens.back() : Value();
      Value readBytes = getReadBytes(rewriter, ctx, loc, remainLen, llMask,
                                     llLen, mask, len, elemBytes);
      createMemOp(rewriter, ctx, loc, startPtr, dstStartPtr, offsetBytes,
                  readBytes, memCpyType);
    }
  }

protected:
  const xpu::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  bool isBf16RoundToMid = false;
  bool isBf16Fast = false;
};

struct XPULoadOpConversion : public ConvertOpToLLVMPattern<triton::xpu::LoadOp>,
                             public LoadStoreConversionBase {
  XPULoadOpConversion(LLVMTypeConverter &converter,
                      const xpu::TargetInfo &targetInfo,
                      ModuleAxisInfoAnalysis &axisAnalysisPass,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  void VecBF16ToFP32Unordered(triton::xpu::LoadOp op, mlir::MLIRContext *ctx,
                              Location &loc,
                              ConversionPatternRewriter &rewriter,
                              Type &resElemTy, int numElems, int resVecSize,
                              int ptrDataVecSize, Value &lmBasePtr,
                              SmallVector<Value> &loadedVals) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType veci16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType veci32Ty = VectorType::get(resVecSize, i32_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(resVecSize, bf16_ty);
    VectorType VecFp16Ty = VectorType::get(ptrDataVecSize, f16_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
    int mask = 0xaaaaaaaa;
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    Value maskNegVal = i32_val(~mask);
    maskNegVal = bitcast(maskNegVal, vec1Ty);
    int16_t pad = 0x8000;
    Value padVec = rewriter.create<LLVM::UndefOp>(loc, veci16Ty);
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      padVec = insert_element(veci16Ty, padVec, i16_val(pad), i16_val(elemIdx));
    }
    for (int i = 0; i < numElems / 2; ++i) {
      Value elemPtr = gep(ptr_ty(ctx, 0), vecBf16Ty, lmBasePtr, i32_val(i));
      Value veven;
      if (isBf16RoundToMid) {
        veven = rewriter.create<mlir::LLVM::XPU::VLOAD_MHOp>(
            loc, veci16Ty, elemPtr, padVec, maskVal);
      } else {
        veven = rewriter.create<mlir::LLVM::XPU::VLOAD_MZOp>(loc, veci16Ty,
                                                             elemPtr, maskVal);
      }
      veven = bitcast(veven, resElemTy);
      loadedVals.emplace_back(veven);
      Value vodd;
      if (isBf16RoundToMid) {
        vodd = rewriter.create<mlir::LLVM::XPU::VLOAD_MHOp>(
            loc, veci16Ty, elemPtr, padVec, maskNegVal);
      } else {
        vodd = rewriter.create<mlir::LLVM::XPU::VLOAD_MZOp>(
            loc, veci16Ty, elemPtr, maskNegVal);
      }
      vodd = bitcast(vodd, VecFp16Ty);
      Value voddSl =
          rewriter.create<mlir::LLVM::XPU::VSHUFFLE2Op>(loc, VecFp16Ty, vodd);
      voddSl = bitcast(voddSl, resElemTy);
      loadedVals.emplace_back(voddSl);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value loaded = load(halfVecBf16Ty, elemPtr);
      loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemTy, loaded);
      loadedVals.emplace_back(loaded);
    }
    return;
  }

  void VecBF16ToFP32(triton::xpu::LoadOp op, mlir::MLIRContext *ctx,
                     Location &loc, ConversionPatternRewriter &rewriter,
                     Type &resElemTy, int numElems, int resVecSize,
                     int ptrDataVecSize, SmallVector<Value> &loadedVals) const {
    VectorType vecFp16Ty = VectorType::get(ptrDataVecSize, f16_ty);
    Value padVec = rewriter.create<LLVM::UndefOp>(loc, vecFp16Ty);
    int16_t pad = isBf16RoundToMid ? 0x8000 : 0;
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      padVec =
          insert_element(vecFp16Ty, padVec, f16_val(pad), i16_val(elemIdx));
    }
    SmallVector<Value> newLoadedVals;
    for (int i = 0; i < numElems / 2; ++i) {
      Value val = bitcast(loadedVals[i], vecFp16Ty);
      Value vl = rewriter.create<mlir::LLVM::XPU::VMERGE_L_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vl = bitcast(vl, resElemTy);
      newLoadedVals.emplace_back(vl);
      Value vh = rewriter.create<mlir::LLVM::XPU::VMERGE_H_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vh = bitcast(vh, resElemTy);
      newLoadedVals.emplace_back(vh);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value ext = rewriter.create<LLVM::FPExtOp>(loc, resElemTy,
                                                 loadedVals[remainedIdx]);
      newLoadedVals.emplace_back(ext);
    }
    loadedVals = newLoadedVals;
    return;
  }

  // Escape hatch for the fence below; on by default because without it the
  // aliased-buffer read is simply wrong (findings 1.74).
  static bool lmWARFenceEnabled() {
    static const bool enabled =
        mlir::triton::tools::isEnvValueBool(
            mlir::triton::tools::getStrEnvXPU("TRITONXPU_LM_WAR_FENCE"))
            .value_or(true);
    return enabled;
  }

  // How many GM2LM-family ops fill the LM buffer this load reads. More than one
  // means MemoryInplace aliased several loads onto the same buffer, which is
  // how a later op comes to rewrite the words being read here. A single filler
  // inside a loop refills the same buffer on the next iteration, which is the
  // same hazard at a longer distance; measured and not reproducible
  // (findings 1.76), because the overwrite either reads the just-loaded
  // register or sits behind the mfence the loop's own store already emits.
  static unsigned countBufferFillers(triton::xpu::LoadOp op) {
    Operation *allocaOp = nullptr;
    if (Operation *def = op.getPtr().getDefiningOp()) {
      if (isa<triton::xpu::AllocaOp>(def))
        allocaOp = def;
      else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(def))
        allocaOp = gm2lmOp.getBufPtr().getDefiningOp();
      else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(def))
        allocaOp = gm2lmOp.getBufPtr().getDefiningOp();
    }
    if (!allocaOp || !isa<triton::xpu::AllocaOp>(allocaOp))
      return 0;
    unsigned fillers = 0;
    for (Operation *user : allocaOp->getUsers())
      if (isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp>(user))
        ++fillers;
    return fillers;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value res = op.getResult();
    Value ptr = op.getPtr();
    Value index = op.getIndex();

    int32_t stride = op.getStride();
    int32_t colSize = op.getTensorColSize();
    bool coreDealMultiRows = colSize != -1;
    bool isDiscreteSame = (stride == 0);
    bool isUnknown = stride != 0 && stride != 1 && !op.getIsDiscrete();
    bool bf16Tofp32Unordered = op.getBf16Tofp32Unordered();

    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llIndex = adaptor.getIndex();

    // Determine Type
    Type ptrTy = ptr.getType();
    Type resTy = res.getType();

    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type resElemTy = typeConverter->convertType(getElementTypeOrSelf(resTy));
    Type ptrDataVecTy = resElemTy;

    unsigned ptrNumElems = getTotalElemsPerThread(ptrTy);
    unsigned resNumElems = getTotalElemsPerThread(resTy);

    Type ptrElemScalarTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      ptrElemScalarTy =
          mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      ptrElemScalarTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }

    Type resElemScalarTy = getElementTypeOrSelf(resElemTy);

    ptrElemScalarTy = typeConverter->convertType(ptrElemScalarTy);
    resElemScalarTy = typeConverter->convertType(resElemScalarTy);

    // Get the LLVM values
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    stride =
        (stride >= 0 && ptrNumElems * stride <= targetInfo.getXPUBufferSize())
            ? stride
            : 1;

    assert(llPtrs.size() == ptrNumElems);
    bool isVectorized = false;
    unsigned vecSize = 1u;
    unsigned elemNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(resElemScalarTy)
            ? 64u
            : resElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(resElemTy)) {
      isVectorized = true;
      getVectorInfo(resElemTy, vecSize, elemNbits);
    }

    // fp16Tofp32
    if (resElemScalarTy.isF32() && ptrElemScalarTy.isF16()) {
      Value fp16LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      Value fp32LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      ValueRange singleOperandRange(
          {fp16LM, fp32LM, i32_val(ptrNumElems * stride)});
      mlir::LLVM::XPU::createDeviceCall("_ZN3xpu10fp16tofp32EPKNS_7float16EPfi",
                                        rewriter, op, singleOperandRange, loc);
      createMfenceLMOp(rewriter, loc);
      ptrElemScalarTy = resElemScalarTy;
    }
    // bf16Tofp32
    bool bf16Tofp32 = false;
    if (resElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
      int ptrVecSize = std::min(ptrNumElems, vecSize * 2);
      ptrDataVecTy = isVectorized ? VectorType::get(ptrVecSize, ptrElemScalarTy)
                                  : ptrElemScalarTy;
      bf16Tofp32 = true;
    }

    unsigned ptrDataVecSize = 1u;
    unsigned ptrDataNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(ptrElemScalarTy)
            ? 64u
            : ptrElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(ptrDataVecTy)) {
      getVectorInfo(ptrDataVecTy, ptrDataVecSize, ptrDataNbits);
    }

    SmallVector<Value> loadedVals;
    Value lmBasePtr = bitcast(llPtrs[0], ptr_ty(ctx, 0));
    if (index) {
      ptrNumElems = resNumElems;
      unsigned _stride =
          (bf16Tofp32 && isVectorized)
              ? std::ceil(static_cast<double>(ptrNumElems * stride) / 2)
              : ptrNumElems * stride;
      _stride = coreDealMultiRows ? stride : _stride;
      Value idx = mul(llIndex, i32_val(_stride));
      lmBasePtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr, idx);
      stride = coreDealMultiRows
                   ? std::ceil(static_cast<double>(colSize) / vecSize)
                   : stride;
    }

    if (op.getSVOpt()) {
      Value elemPtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value loaded = load(ptrElemScalarTy, elemPtr);
      if (bf16Tofp32) {
        loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
      }
      loadedVals.push_back(loaded);
    } else if (isDiscreteSame) {
      Value elemPtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value loaded = load(ptrElemScalarTy, elemPtr);
      if (bf16Tofp32) {
        loaded = rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
      }
      for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
        if (isVectorized) {
          Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t idx = 0; idx < vecSize; ++idx) {
            newVector =
                insert_element(resElemTy, newVector, loaded, i32_val(idx));
          }
          loadedVals.push_back(newVector);
        } else {
          loadedVals.push_back(loaded);
        }
      }
    } else if (op.getIsDiscrete()) {
      if (isVectorized) {
        for (size_t vecIdx = 0; vecIdx < resNumElems; ++vecIdx) {
          Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
          for (size_t elemIdx = 0; elemIdx < vecSize; ++elemIdx) {
            auto idx = vecIdx * vecSize + elemIdx;
            Value elemPtr = bitcast(llPtrs[idx], ptr_ty(ctx, 0));
            Value loaded = load(ptrElemScalarTy, elemPtr);
            if (bf16Tofp32) {
              loaded =
                  rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
            }
            // insert val to newVector
            newVector =
                insert_element(resElemTy, newVector, loaded, i32_val(elemIdx));
          }
          loadedVals.push_back(newVector);
        }
      } else {
        for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
          Value elemPtr = bitcast(llPtrs[elemIdx], ptr_ty(ctx, 0));
          Value loaded = load(ptrElemScalarTy, elemPtr);
          if (bf16Tofp32) {
            loaded =
                rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
          }
          loadedVals.push_back(loaded);
        }
      }
    } else if (!(coreDealMultiRows && index) && stride > 1 && isVectorized &&
               ptrDataVecSize * ptrDataNbits == 512) {
      // Vgather
      VectorType offsetTy =
          VectorType::get(ptrDataVecSize, int_ty(ptrDataNbits));
      Value offsetVec = rewriter.create<LLVM::UndefOp>(loc, offsetTy);
      for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
        Value offsetVal =
            int_val(ptrDataNbits, (ptrDataNbits / 8u) * stride * elemIdx);
        offsetVec = insert_element(offsetTy, offsetVec, offsetVal,
                                   int_val(ptrDataNbits, elemIdx));
      }
      Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      for (size_t vecIdx = 0;
           vecIdx < (index ? ptrNumElems : (ptrNumElems / ptrDataVecSize));
           ++vecIdx) {
        Value vecPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, _lmBasePtr,
                           int_val(ptrDataNbits, vecIdx * stride));
        Value tmpPtr = bitcast(vecPtr, ptr_ty(ctx, 0));
        Value vgather;
        if (ptrElemScalarTy.isF32() || ptrElemScalarTy.isInteger(32)) {
          vgather = rewriter.create<mlir::LLVM::XPU::VGatherFOp>(
              loc, offsetTy, tmpPtr, offsetVec);
        } else if (ptrElemScalarTy.isF16() || ptrElemScalarTy.isBF16() ||
                   ptrElemScalarTy.isInteger(16)) {
          vgather = rewriter.create<mlir::LLVM::XPU::VGatherHFOp>(
              loc, offsetTy, tmpPtr, offsetVec);
        } else {
          llvm_unreachable("Only support I16/FP16/BF16/I32/FP32 in VGather!");
        }
        Value loaded = bitcast(vgather, resElemTy);
        loadedVals.push_back(loaded);
      }
      if (bf16Tofp32) {
        VecBF16ToFP32(op, ctx, loc, rewriter, resElemTy, resNumElems, vecSize,
                      ptrDataVecSize, loadedVals);
      }
    } else {
      if (isVectorized) {
        if (bf16Tofp32) {
          if (bf16Tofp32Unordered) {
            VecBF16ToFP32Unordered(op, ctx, loc, rewriter, resElemTy,
                                   resNumElems, vecSize, ptrDataVecSize,
                                   lmBasePtr, loadedVals);
          } else {
            Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
            for (size_t elemIdx = 0; elemIdx < resNumElems / 2; elemIdx++) {
              Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, _lmBasePtr,
                                  i32_val(elemIdx * stride));
              Value loaded = load(ptrDataVecTy, elemPtr);
              loadedVals.push_back(loaded);
            }
            int remainedIdx = 2 * (resNumElems / 2);
            if (resNumElems - remainedIdx) {
              VectorType halfVecBf16Ty = VectorType::get(vecSize, bf16_ty);
              Value elemPtr = gep(ptr_ty(ctx, 0), halfVecBf16Ty, _lmBasePtr,
                                  i32_val(remainedIdx * stride));
              Value loaded = load(halfVecBf16Ty, elemPtr);
              loadedVals.push_back(loaded);
            }
            VecBF16ToFP32(op, ctx, loc, rewriter, resElemTy, resNumElems,
                          vecSize, ptrDataVecSize, loadedVals);
          }
        } else {
          Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
          for (size_t elemIdx = 0; elemIdx < resNumElems; elemIdx++) {
            Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, _lmBasePtr,
                                i32_val(elemIdx * stride));
            Value loaded = load(ptrDataVecTy, elemPtr);
            loadedVals.push_back(loaded);
          }
        }
      } else {
        Value _lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
        for (size_t elemIdx = 0; elemIdx < ptrNumElems; elemIdx++) {
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, _lmBasePtr,
                              i32_val(elemIdx * stride));
          Value loaded = load(ptrElemScalarTy, elemPtr);
          if (bf16Tofp32) {
            loaded =
                rewriter.create<LLVM::FPExtOp>(loc, resElemScalarTy, loaded);
          }
          loadedVals.push_back(loaded);
        }
      }
    }

    // Write-after-read hazard on an *aliased* LM buffer (findings 1.74).
    // MemoryInplace folds allocas whose IR lifetimes are disjoint into one
    // buffer, so the words this load just read get rewritten a few instructions
    // later by the next load's other-fill. On xpu3 that scalar store beats the
    // in-flight `vload_mask16` to word 0 of the buffer, per core and
    // nondeterministically: a vectorized three-operand welford reduce lost the
    // lane-0 contribution of ~55 of 64 cores, while the third operand -- the
    // one whose buffer is never rewritten afterwards -- was always intact.
    // Fence the reads so the buffer is free to be rewritten. Only a buffer with
    // more than one filler can be in that situation, and fencing just those
    // leaves every other kernel's code byte-identical.
    if (lmWARFenceEnabled() && countBufferFillers(op) > 1)
      createMfenceLMOp(rewriter, loc);

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPULoadScalarIndexedOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::LoadScalarIndexedOp>,
      public LoadStoreConversionBase {
  XPULoadScalarIndexedOpConversion(LLVMTypeConverter &converter,
                                   const xpu::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LoadScalarIndexedOp>(converter,
                                                                 benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::LoadScalarIndexedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value res = op.getResult();
    Type resTy = res.getType();
    Type resElemTy = typeConverter->convertType(getElementTypeOrSelf(resTy));
    Type resElemScalarTy = getElementTypeOrSelf(resElemTy);
    resElemScalarTy = typeConverter->convertType(resElemScalarTy);

    unsigned addrSpace = 0;
    Type ptrTy = op.getPtr().getType();
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      if (auto pt =
              mlir::dyn_cast<triton::PointerType>(ptrTensorTy.getElementType()))
        addrSpace = pt.getAddressSpace();
    } else if (auto pt = mlir::dyn_cast<triton::PointerType>(ptrTy)) {
      addrSpace = pt.getAddressSpace();
    }

    auto llPtrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    Value basePtr = bitcast(llPtrs[0], ptr_ty(ctx, addrSpace));
    Value llIndex = adaptor.getIndex();
    Value elemPtr =
        gep(ptr_ty(ctx, addrSpace), resElemScalarTy, basePtr, llIndex);
    Value loaded = load(resElemScalarTy, elemPtr);

    unsigned resNumElems = getTotalElemsPerThread(resTy);
    bool isVectorized = mlir::isa<mlir::VectorType>(resElemTy);
    unsigned vecSize = 1u;
    if (isVectorized) {
      unsigned elemNbits = 0u;
      getVectorInfo(resElemTy, vecSize, elemNbits);
    }

    SmallVector<Value> loadedVals;
    for (size_t elemIdx = 0; elemIdx < resNumElems; ++elemIdx) {
      if (isVectorized) {
        Value newVector = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
        for (size_t i = 0; i < vecSize; ++i) {
          newVector = insert_element(resElemTy, newVector, loaded, i32_val(i));
        }
        loadedVals.push_back(newVector);
      } else {
        loadedVals.push_back(loaded);
      }
    }

    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::StoreOp>,
      public LoadStoreConversionBase {
  XPUStoreOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  void VecFP32ToBF16Unordered(triton::xpu::StoreOp op, mlir::MLIRContext *ctx,
                              Location &loc,
                              ConversionPatternRewriter &rewriter, int numElems,
                              int valueVecSize, int ptrDataVecSize,
                              SmallVector<Value> &valueElems,
                              Value &lmBasePtr) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType veci16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType veci32Ty = VectorType::get(valueVecSize, i32_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // vecBf16Ty
    int mask = 0xaaaaaaaa;
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    Value maskNegVal = i32_val(~mask);
    maskNegVal = bitcast(maskNegVal, vec1Ty);
    Value poseVal = i32_val(16);
    uint32_t one = 0x0001;
    uint32_t magic = 0x7fff;
    for (int i = 0; i < numElems / 2; ++i) {
      Value veven = bitcast(valueElems[2 * i], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> vevenAndOperands({i32_val(one), veven});
        auto vevenAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vevenAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> evenOperands({i32_val(magic), vevenAnd.getRes()});
        auto vevenSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, evenOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        veven = add(veven, vevenSvAdd.getRes());
      }
      veven = bitcast(veven, veci16Ty);
      Value elemPtr = gep(ptr_ty(ctx, 0), vecBf16Ty, lmBasePtr, i32_val(i));
      rewriter.create<mlir::LLVM::XPU::VSTORE_MHOp>(loc, veven, elemPtr,
                                                    maskVal);
      Value vodd = bitcast(valueElems[2 * i + 1], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> oddAndOperands({i32_val(one), vodd});
        auto voddAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, oddAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> oddOperands({i32_val(magic), voddAnd.getRes()});
        auto voddSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, oddOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        vodd = add(vodd, voddSvAdd.getRes());
      }
      Value voddSr = rewriter.create<mlir::LLVM::XPU::SVSRLPOp>(loc, veci32Ty,
                                                                poseVal, vodd);
      voddSr = bitcast(voddSr, veci16Ty);
      rewriter.create<mlir::LLVM::XPU::VSTORE_MHOp>(loc, voddSr, elemPtr,
                                                    maskNegVal);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

  void VecFP32ToBF16Slow(triton::xpu::StoreOp op, mlir::MLIRContext *ctx,
                         Location &loc, ConversionPatternRewriter &rewriter,
                         int numElems, int valueVecSize, int ptrDataVecSize,
                         SmallVector<Value> &valueElems,
                         Value &lmBasePtr) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType vecI16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    VectorType veci32Ty = VectorType::get(valueVecSize, i32_ty);
    constexpr int mask = 0xaaaaaaaa; // 0b10101010101010101010101010101010
    Value maskVal = i32_val(mask);
    maskVal = bitcast(maskVal, vec1Ty);
    SmallVector<int16_t> offset_v = {0,  0,  0,  2,  0,  4,  0,  6,  0,  8, 0,
                                     10, 0,  12, 0,  14, 0,  16, 0,  18, 0, 20,
                                     0,  22, 0,  24, 0,  26, 0,  28, 0,  30};
    Value offsetVec = rewriter.create<LLVM::UndefOp>(loc, vecI16Ty);
    for (size_t elemIdx = 0; elemIdx < ptrDataVecSize; ++elemIdx) {
      offsetVec = insert_element(vecI16Ty, offsetVec,
                                 i16_val(offset_v[elemIdx]), i16_val(elemIdx));
    }
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // halfVecBf16Ty
    uint32_t one = 0x0001;
    uint32_t magic = 0x7fff;
    for (int i = 0; i < numElems / 2; ++i) {
      Value dstPtr1 =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i16_val(2 * i));
      Value vl = bitcast(valueElems[2 * i], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> vlAndOperands({i32_val(one), vl});
        auto vlAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vlAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> vlOperands({i32_val(magic), vlAnd.getRes()});
        auto vlSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vlOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        vl = add(vl, vlSvAdd.getRes());
      }
      vl = bitcast(vl, vecI16Ty);
      rewriter.create<mlir::LLVM::XPU::SCATTER_MHOp>(loc, vl, maskVal, dstPtr1,
                                                     offsetVec);
      Value vh = bitcast(valueElems[2 * i + 1], veci32Ty);
      if (!isBf16RoundToMid) {
        SmallVector<Value, 4> vhAndOperands({i32_val(one), vh});
        auto vhAnd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vhAndOperands, "vand.u.mz $0{mr1}, $1, $2",
            "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        SmallVector<Value, 4> vhOperands({i32_val(magic), vhAnd.getRes()});
        auto vhSvAdd = rewriter.create<LLVM::InlineAsmOp>(
            loc, veci32Ty, vhOperands, "vadd.u.mz $0{mr1}, $1, $2", "=&v,r,v",
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        vh = add(vh, vhSvAdd.getRes());
      }
      vh = bitcast(vh, vecI16Ty);
      Value dstPtr2 =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i16_val(2 * i + 1));
      rewriter.create<mlir::LLVM::XPU::SCATTER_MHOp>(loc, vh, maskVal, dstPtr2,
                                                     offsetVec);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

  void VecFP32ToBF16(triton::xpu::StoreOp op, mlir::MLIRContext *ctx,
                     Location &loc, ConversionPatternRewriter &rewriter,
                     int numElems, int valueVecSize, int ptrDataVecSize,
                     SmallVector<Value> &valueElems, Value &lmBasePtr) const {
    VectorType vecBf16Ty = VectorType::get(ptrDataVecSize, bf16_ty);
    VectorType vecI16Ty = VectorType::get(ptrDataVecSize, i16_ty);
    VectorType vec1Ty = VectorType::get(ptrDataVecSize, i1_ty);
    VectorType halfVecBf16Ty = VectorType::get(valueVecSize, bf16_ty);
    VectorType veci32Ty = VectorType::get(valueVecSize, i32_ty);
    lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0)); // halfVecBf16Ty
    for (int i = 0; i < numElems / 2; ++i) {
      Value dstPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(2 * i));
      ValueRange args({dstPtr, valueElems[2 * i], valueElems[2 * i + 1]});
      LLVM::XPU::createDeviceCall("_ZN3xpu10vstore2_lmEPNS_8bfloat16EDv16_fS2_",
                                  rewriter, op, args, loc);
    }
    if (numElems % 2 == 1) {
      int remainedIdx = numElems - 1;
      Value elemPtr =
          gep(ptr_ty(ctx, 0), halfVecBf16Ty, lmBasePtr, i32_val(remainedIdx));
      Value elem = valueElems[remainedIdx];
      Value trunc = rewriter.create<LLVM::FPTruncOp>(loc, halfVecBf16Ty, elem);
      store(trunc, elemPtr);
    }
    return;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value index = op.getIndex();

    int32_t colSize = op.getTensorColSize();
    bool coreDealMultiRows = colSize != -1;
    bool bf16Tofp32Unordered = op.getBf16Tofp32Unordered();
    auto dtype = op.getDtype();

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llValue = adaptor.getValue();
    Value llIndex = adaptor.getIndex();

    // Determine Type
    Type ptrTy = ptr.getType();
    auto valueTy = value.getType();

    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type valueElemScalarTy = getElementTypeOrSelf(valueElemTy);
    Type ptrElemScalarTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      ptrElemScalarTy =
          mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      ptrElemScalarTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    ptrElemScalarTy = typeConverter->convertType(ptrElemScalarTy);
    valueElemScalarTy = typeConverter->convertType(valueElemScalarTy);

    Type ptrDataVecTy = valueElemTy;

    unsigned valueNumElems = getTotalElemsPerThread(valueTy);
    unsigned ptrNumElems = getTotalElemsPerThread(ptrTy);

    // Get the LLVM values
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    auto llVals = unpackLLElements(loc, llValue, rewriter);
    // Determine the vectorization size
    bool isVectorized = mlir::isa<mlir::VectorType>(valueElemTy);
    unsigned valueVecSize = 1u;
    unsigned valueScalarNbits = 32u;
    if (mlir::isa<mlir::VectorType>(valueElemTy)) {
      isVectorized = true;
      getVectorInfo(valueElemTy, valueVecSize, valueScalarNbits);
    }

    // fp32 to bf16
    bool fp32Tobf16 = false;
    if (valueElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
      int ptrVecSize = std::min(ptrNumElems, valueVecSize * 2);
      ptrDataVecTy = isVectorized ? VectorType::get(ptrVecSize, ptrElemScalarTy)
                                  : ptrElemScalarTy;
      fp32Tobf16 = true;
    }

    bool fp32Tofp16 = false;
    if (valueElemScalarTy.isF32() && ptrElemScalarTy.isF16()) {
      ptrElemScalarTy = valueElemScalarTy;
      fp32Tofp16 = true;
    }

    unsigned ptrDataVecSize = 1u;
    unsigned ptrDataNbits =
        isa<triton::PointerType, LLVM::LLVMPointerType>(ptrElemScalarTy)
            ? 64u
            : ptrElemScalarTy.getIntOrFloatBitWidth();
    if (mlir::isa<mlir::VectorType>(ptrDataVecTy)) {
      getVectorInfo(ptrDataVecTy, ptrDataVecSize, ptrDataNbits);
    }

    Value lmBasePtr = bitcast(llPtrs[0], ptr_ty(ctx, 0));
    unsigned stride = 1;
    if (index) {
      ptrNumElems = valueNumElems;
      unsigned _stride = (fp32Tobf16 && isVectorized)
                             ? std::ceil(static_cast<double>(ptrNumElems) / 2)
                             : ptrNumElems;
      _stride = coreDealMultiRows ? 1 : _stride;
      if (valueElemScalarTy.isInteger(32) && ptrElemScalarTy.isInteger(8)) {
        valueVecSize = dtype == Dtype::FP32 ? 16 : 32;
        Value idx = mul(llIndex, i32_val(valueNumElems * valueVecSize));
        lmBasePtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr, idx);
      } else {
        Value idx = mul(llIndex, i32_val(_stride));
        lmBasePtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr, idx);
      }
      stride = coreDealMultiRows
                   ? std::ceil(static_cast<double>(colSize) / ptrDataVecSize)
                   : stride;
    }

    if (valueElemScalarTy.isInteger(32) && ptrElemScalarTy.isInteger(8)) {
      lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
      if (dtype == Dtype::FP32) {
        for (int i = 0; i < valueNumElems; i += 4) {
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                              i32_val(i * valueVecSize * stride));
          ValueRange args({llVals[i], llVals[i + 1], llVals[i + 2],
                           llVals[i + 3], elemPtr});
          if (valueElemScalarTy.isUnsignedInteger()) {
            LLVM::XPU::createDeviceCall("_ZN3xpu8vstorei8EjjjjPa", rewriter, op,
                                        args, loc);
          } else if (valueElemScalarTy.isInteger(32)) {
            LLVM::XPU::createDeviceCall("_ZN3xpu12vstorei8_i32EiiiiPa",
                                        rewriter, op, args, loc);
          } else if (valueElemScalarTy.isInteger(64)) {
            LLVM::XPU::createDeviceCall("_ZN3xpu12vstorei8_i64EllllPa",
                                        rewriter, op, args, loc);
          }
        }
      } else if (dtype == Dtype::FP16) {
        for (int i = 0; i < valueNumElems; i += 2) {
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                              i32_val(i * valueVecSize * stride));
          ValueRange args({llVals[i], llVals[i + 1], elemPtr});
          LLVM::XPU::createDeviceCall("_ZN3xpu16vstorei8_unroll2EjjPa",
                                      rewriter, op, args, loc);
        }
      } else {
        llvm_unreachable("vstorei8 only supports FP32 or FP16");
      }
    } else {
      if (isVectorized) {
        if (valueElemScalarTy.isF32() && ptrElemScalarTy.isBF16()) {
          if (bf16Tofp32Unordered) {
            VecFP32ToBF16Unordered(op, ctx, loc, rewriter, valueNumElems,
                                   valueVecSize, ptrDataVecSize, llVals,
                                   lmBasePtr);
          } else {
            if (isBf16Fast) {
              VecFP32ToBF16(op, ctx, loc, rewriter, valueNumElems, valueVecSize,
                            ptrDataVecSize, llVals, lmBasePtr);
            } else {
              VecFP32ToBF16Slow(op, ctx, loc, rewriter, valueNumElems,
                                valueVecSize, ptrDataVecSize, llVals,
                                lmBasePtr);
            }
          }
        } else {
          lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
          for (size_t elemIdx = 0; elemIdx < valueNumElems; elemIdx++) {
            Value elem = llVals[elemIdx];
            Value elemPtr = gep(ptr_ty(ctx, 0), ptrDataVecTy, lmBasePtr,
                                i32_val(elemIdx * stride));
            elem = bf16ToFp16(rewriter, loc, valueElemTy, elem);
            store(elem, elemPtr);
          }
        }
      } else {
        lmBasePtr = bitcast(lmBasePtr, ptr_ty(ctx, 0));
        for (size_t elemIdx = 0; elemIdx < ptrNumElems; elemIdx++) {
          Value elem = llVals[elemIdx];
          if (fp32Tobf16) {
            elem = rewriter.create<LLVM::FPTruncOp>(loc, ptrElemScalarTy, elem);
          }
          Value elemPtr = gep(ptr_ty(ctx, 0), ptrElemScalarTy, lmBasePtr,
                              i32_val(elemIdx * stride));
          store(elem, elemPtr);
        }
      }
    }

    createMfenceLMOp(rewriter, loc);

    // fp32 to fp16
    if (fp32Tofp16) {
      Value fp16LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      Value fp32LM = bitcast(llPtrs[0], ptr_ty(ctx, 0));
      ValueRange singleOperandRange({fp32LM, fp16LM, i32_val(ptrNumElems)});
      mlir::LLVM::XPU::createDeviceCall("_ZN3xpu10fp32tofp16Ef", rewriter, op,
                                        singleOperandRange, loc);
      createMfenceLMOp(rewriter, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct XPUAllocaOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::AllocaOp>,
      public LoadStoreConversionBase {
  XPUAllocaOpConversion(LLVMTypeConverter &converter,
                        const xpu::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::AllocaOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    auto resTy = op.getType();
    Type valueElemTy;
    if (auto resTensorTy = mlir::dyn_cast<RankedTensorType>(resTy)) {
      // Tensor
      valueElemTy =
          mlir::cast<triton::PointerType>(resTensorTy.getElementType())
              .getPointeeType();
    } else {
      // Scalar
      valueElemTy = mlir::cast<triton::PointerType>(resTy).getPointeeType();
    }
    valueElemTy = typeConverter->convertType(valueElemTy);

    unsigned numElems = getTotalElemsPerThread(resTy);

    auto allocNumElems = numElems;
    if (static_cast<XPUArch>(targetInfo.getXPUArch()) == XPUArch::XPU2 &&
        valueElemTy.isF16()) {
      // algin to 32, cause fp16tofp32 use vector<32*fp16> instruction
      allocNumElems = (allocNumElems + 31) / 32 * 32;
      // double space to accommodate 32*fp32
      allocNumElems *= 2;
    }
    for (auto user : op->getUsers()) {
      if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
        auto fixedStride = gm2lmOp.getFixedStride();
        if (fixedStride > 0 &&
            fixedStride * numElems <= targetInfo.getXPUBufferSize()) {
          allocNumElems *= fixedStride;
        }
      } else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(user)) {
        auto fixedStride = gm2lmOp.getFixedStride();
        if (fixedStride > 0 &&
            fixedStride * numElems <= targetInfo.getXPUBufferSize()) {
          allocNumElems *= fixedStride;
        }
      }
    }

    allocNumElems =
        align(allocNumElems, valueElemTy, 64); // 64 bytes aligned for LM
    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    auto lmBuf = allocate(lmPtrTy, valueElemTy, i32_val(allocNumElems));

    SmallVector<Value> lmPtrs;
    for (int i = 0; i < numElems; i++) {
      lmPtrs.push_back(gep(lmPtrTy, valueElemTy, lmBuf, i32_val(i)));
    }
    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Value resultStruct = packLLElements(loc, typeConverter, lmPtrs, rewriter,
                                        llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUGM2LMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GM2LMOp>,
      public LoadStoreConversionBase {
  XPUGM2LMOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GM2LMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GM2LMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value len = op.getLen();
    Value res = op.getResult();
    int32_t tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llLen = adaptor.getLen();
    Value llGMPtr = adaptor.getPtr();
    Value llLMPtr = adaptor.getBufPtr();
    Value resultStruct = llLMPtr;

    Type ptrTy = ptr.getType();
    Type resTy = res.getType();
    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    assert(llLMPtr && "llBufPtr should not be null.");
    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);

    Value elemBytes = i32_val(elemNbits / 8u);
    Value offsetBytes = i32_val(0);

    bool mask = false;
    unsigned lenElemBit = 32;
    llvm::SmallVector<Value> llLens;
    if (op.getLen()) {
      auto lenElemTy = getElementTypeOrSelf(op.getLen().getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      mask = llGMPtrs.size() > 1 ? mlir::isa<mlir::IntegerType>(lenElemTy) &&
                                       (lenElemBit == 32 || lenElemBit == 64)
                                 : false;
    }

    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (mask) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);

    OffsetState offsetState = static_cast<OffsetState>(op.getOffsetState());
    int32_t fixedStride = op.getFixedStride();
    if (offsetState == OffsetState::Unknown) {
      /*  Small Col Size Opt Mask(14 < 16)

          Before Opt:
              T T T T T T T T
              T T T T T T F F

          After Opt:
              T T T T T T T F
              T T T T T T T F
      */
      SmallVector<bool> maskLists;
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        auto tensorRowSize =
            std::ceil(static_cast<double>(shape[0]) / 64);   // 128 / 64 = 2
        auto memColSize = shape[1];                          // 16
        unsigned rowRemainElem = memColSize - tensorColSize; // 16 - 15 = 1

        for (size_t row_idx = 0; row_idx < tensorRowSize; ++row_idx) {
          for (size_t col_idx = 0; col_idx < tensorColSize; ++col_idx) {
            maskLists.push_back(true);
          }

          for (size_t remainElem = rowRemainElem; remainElem > 0;
               --remainElem) {
            maskLists.push_back(false);
          }
        }
      }

      if (fixedStride > 0 &&
          numElems * fixedStride <= targetInfo.getXPUBufferSize()) {
        // Unknown FixedStride Vgather
        readBytes = mul(i32_val(fixedStride), readBytes);
      } else {
        // Unknown
        readBytes = elemBytes;
        for (size_t i = 0; i < llGMPtrs.size(); ++i) {
          //  Protect Ptr Boundary Condition
          Value base;
          if (coreDealMultiRows) {
            base = mask ? select(int_val(1, maskLists[i]), llGMPtrs[i],
                                 llGMPtrs[0])
                        : llGMPtrs[i];
          } else {
            if (mask) { // Has Mask
              if (llLens[0].getType().isInteger(32)) {
                base = select(icmp_slt(i32_val(i), llLens[0]), llGMPtrs[i],
                              llGMPtrs[0]);
              } else if (llLens[0].getType().isInteger(64)) {
                base = select(icmp_slt(i64_val(i), llLens[0]), llGMPtrs[i],
                              llGMPtrs[0]);
              } else {
                llvm_unreachable("Unsupported Mask Int Type");
              }
            } else {
              base = llGMPtrs[i];
            }
          }
          Value dstPtr = bitcast(llLMPtrs[i], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(base, ptr_ty(ctx, 1));
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
          if (!async)
            createMfenceLMOp(rewriter, loc);
        }

        resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                      llvmResultStructTy);
        rewriter.replaceOp(op, {resultStruct});
        return success();
      }
    } else if (offsetState == OffsetState::Discrete) {
      // Reorder the local buffer ptrs.
      SmallVector<Value> newLmBufPtrs(llGMPtrs.size());
      Value basePtrInt = ptrtoint(i64_ty, llGMPtrs[0]);
      for (size_t idx = 0; idx < llGMPtrs.size(); ++idx) {
        Value elemPtrInt = ptrtoint(i64_ty, llGMPtrs[idx]); // convert to int
        Value offsetBytes =
            sub(elemPtrInt, basePtrInt); // get the offset(Bytes)
        Value elemPtr = gep(ptr_ty(ctx, 0), i8_ty, llLMPtrs[0], offsetBytes);
        newLmBufPtrs[idx] = elemPtr;
      }
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::DiscreteSame) {
      readBytes = elemBytes;
      SmallVector<Value> newLmBufPtrs(llLMPtrs.size(), llLMPtrs[0]);
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llLMPtr.getType());
    } else if (offsetState == OffsetState::LocallyScalar) {
      int64_t rowLen = op.getRowLen();
      if (rowLen <= 0)
        rowLen = static_cast<int64_t>(llLMPtrs.size());
      readBytes = elemBytes;
      SmallVector<Value> newLmBufPtrs(llLMPtrs.size());
      for (size_t start = 0; start < llLMPtrs.size();
           start += static_cast<size_t>(rowLen)) {
        Value dstPtr = bitcast(llLMPtrs[start], ptr_ty(ctx, 0));
        Value srcPtr = bitcast(llGMPtrs[start], ptr_ty(ctx, 1));
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
        size_t end = start + static_cast<size_t>(rowLen);
        if (end > llLMPtrs.size())
          end = llLMPtrs.size();
        for (size_t j = start; j < end; ++j)
          newLmBufPtrs[j] = llLMPtrs[start];
      }
      if (!async)
        createMfenceLMOp(rewriter, loc);
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
      rewriter.replaceOp(op, {resultStruct});
      return success();
    } else if (offsetState == OffsetState::LocallyContinuous) {
      int64_t _rowLen = op.getRowLen();
      int64_t _rowStride = op.getRowStride();
      if (_rowLen % numElems == 0) {
        offsetState = OffsetState::Continuous;
        LLVM_DEBUG(llvm::dbgs() << "[OffsetState]: GM2LM Update "
                                   "LocallyContinuous to Continuous\n");
      } else {
        auto oldBlock = op->getBlock();
        auto newBlock = oldBlock->splitBlock(op->getNextNode());
        int64_t _elemBytes = elemNbits / 8u;
        int64_t _bufLen = static_cast<int64_t>(numElems);
        LLVM_DEBUG(llvm::dbgs() << "[GM2LM LocallyContinuous]: rowLen is "
                                << _rowLen << ", rowStride is " << _rowStride
                                << ", bufLen is " << _bufLen << "\n");
        if (_rowStride == -1) {
          lowerLocallyContinuousUnfixedStride(
              op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llGMPtr, llLMPtr,
              llLen, offsetBytes, MemCpyType::GM2LM, oldBlock, newBlock);
        } else {
          if (_rowLen > _bufLen) {
            lowerLocallyContinuousLargeRow(
                op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM, oldBlock,
                newBlock);
          } else {
            lowerLocallyContinuousSmallRow(
                op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM, oldBlock,
                newBlock);
          }
        }

        if (!async)
          createMfenceLMOp(rewriter, loc);

        resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                      llvmResultStructTy);
        rewriter.replaceOp(op, {resultStruct});
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
    }

    if (coreDealMultiRows) {
      auto shape = cast<RankedTensorType>(ptrTy).getShape();
      int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
      if (tensorColSize % shape[1] == 0) {
        readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
        Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
        Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
      } else {
        readBytes = mul(i32_val(tensorColSize), elemBytes);
        for (int i = 0; i < tensorRowSize; ++i) {
          Value dstPtr = bitcast(llLMPtrs[i * shape[1]], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(llGMPtrs[i * tensorColSize], ptr_ty(ctx, 1));
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        }
      }

      if (!async)
        createMfenceLMOp(rewriter, loc);

      rewriter.replaceOp(op, {resultStruct});
      return success();
    }

    Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
    Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
    createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes, readBytes);
    if (!async)
      createMfenceLMOp(rewriter, loc);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPUStageSMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::StageSMOp>,
      public LoadStoreConversionBase {
  XPUStageSMOpConversion(LLVMTypeConverter &converter,
                         const xpu::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::StageSMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  Value getGlobalSmemBase(Location loc, ConversionPatternRewriter &rewriter,
                          Operation *op) const {
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    LLVM::GlobalOp globalSmem;
    mod.walk([&](LLVM::GlobalOp g) {
      if (g.getSymName() == "global_smem")
        globalSmem = g;
    });
    assert(globalSmem && "global_smem not found; initSharedMemory must run "
                         "before StageSM lowering");
    Value addr = rewriter.create<LLVM::AddressOfOp>(loc, globalSmem);
    return rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 2), addr);
  }

  LogicalResult
  matchAndRewrite(triton::xpu::StageSMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value ptr = op.getPtr();
    Type ptrTy = ptr.getType();
    Type elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    elemTy = typeConverter->convertType(elemTy);
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();

    Value srcPtr = bitcast(adaptor.getPtr(), ptr_ty(ctx, 1));
    Value smBase = getGlobalSmemBase(loc, rewriter, op);
    Value smDst = gep(ptr_ty(ctx, 2), i8_ty, smBase, adaptor.getSmOffset());

    Value elemBytes = i32_val(elemNbits / 8u);
    Value bufElems = adaptor.getBufElems();
    Value readLen = bufElems;
    if (op.getLen()) {
      Value reqLen = smax(adaptor.getLen(), i32_val(0));
      readLen = smin(reqLen, bufElems);
    }
    Value readBytes = mul(readLen, elemBytes);

    Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
    Value isCore0 = icmp_eq(coreId, i32_val(0));

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *dmaBlock = rewriter.createBlock(afterBlock);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, isCore0, dmaBlock, afterBlock);

    rewriter.setInsertionPointToStart(dmaBlock);
    createGM2SMOp(rewriter, ctx, loc, srcPtr, smDst, i32_val(0), readBytes);
    rewriter.create<LLVM::BrOp>(loc, afterBlock);

    rewriter.setInsertionPointToStart(afterBlock);
    xpu_barrier();
    rewriter.replaceOp(op, {smDst});
    return success();
  }
};

struct XPULM2GMOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::LM2GMOp>,
      public LoadStoreConversionBase {

  XPULM2GMOpConversion(LLVMTypeConverter &converter,
                       const xpu::TargetInfo &targetInfo,
                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LM2GMOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::LM2GMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value len = op.getLen();
    int32_t offsetStateInt = op.getOffsetState();
    OffsetState offsetState = static_cast<OffsetState>(offsetStateInt);
    auto tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    offsetState = (tensorColSize == 1) ? OffsetState::Continuous : offsetState;

    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llLen = adaptor.getLen();
    Value llBufPtr = adaptor.getBufPtr();
    assert(llBufPtr && "llBufPtr should not be null.");

    // Get elemTy and numElems
    Type ptrTy = ptr.getType();
    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    Value elemBytes = i32_val(elemNbits / 8u);
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    // Get base, readBytes and offsetBytes
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);

    Value base = llPtrs[0];
    Value offsetBytes = i32_val(0);

    llvm::SmallVector<Value> llLens;
    bool mask = false;
    unsigned lenElemBit = 32;
    if (op.getLen()) {
      auto lenElemTy = getElementTypeOrSelf(op.getLen().getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      mask = llPtrs.size() > 1 ? mlir::isa<mlir::IntegerType>(lenElemTy) &&
                                     (lenElemBit == 32 || lenElemBit == 64)
                               : false;
    }
    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (mask) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);
    auto lmBufPtrs = unpackLLElements(loc, llBufPtr, rewriter);
    Value lmBuf = lmBufPtrs[0];

    // Create LM2GM and mfence
    switch (offsetState) {
    case OffsetState::Continuous: {
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
        if (tensorColSize % shape[1] == 0) {
          readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
          Value srcPtr = bitcast(lmBufPtrs[0], ptr_ty(ctx, 0));
          Value dstPtr = bitcast(llPtrs[0], ptr_ty(ctx, 1));
          createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        } else {
          for (int i = 0; i < tensorRowSize; ++i) {
            readBytes = mul(i32_val(tensorColSize), elemBytes);
            Value srcPtr = bitcast(lmBufPtrs[i * shape[1]], ptr_ty(ctx, 0));
            Value dstPtr = bitcast(llPtrs[i * tensorColSize], ptr_ty(ctx, 1));
            createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                          readBytes);
          }
        }

      } else {
        Value srcPtr = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value basePtr = bitcast(base, ptr_ty(ctx, 1));
        createLM2GMOp(rewriter, ctx, loc, srcPtr, basePtr, offsetBytes,
                      readBytes);
      }
      break;
    }
    case OffsetState::LocallyContinuous: {
      int64_t _rowLen = op.getRowLen();
      int64_t _rowStride = op.getRowStride();
      if (_rowLen % numElems == 0) {
        offsetState = OffsetState::Continuous;
        LLVM_DEBUG(llvm::dbgs() << "[OffsetState]: LM2GM Update "
                                   "LocallyContinuous to Continuous\n");
      } else {
        auto oldBlock = op->getBlock();
        auto newBlock = oldBlock->splitBlock(op->getNextNode());
        int64_t _elemBytes = elemNbits / 8u;
        int64_t _bufLen = static_cast<int64_t>(numElems);
        LLVM_DEBUG(llvm::dbgs() << "[LM2GM LocallyContinuous]: rowLen is "
                                << _rowLen << ", rowStride is " << _rowStride
                                << ", bufLen is " << _bufLen << "\n");

        if (_rowStride == -1) {
          lowerLocallyContinuousUnfixedStride(
              op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llPtr, llBufPtr,
              llLen, offsetBytes, MemCpyType::LM2GM, oldBlock, newBlock);
        } else {
          if (_rowLen > _bufLen) {
            lowerLocallyContinuousLargeRow(
                op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM, oldBlock,
                newBlock);
          } else {
            lowerLocallyContinuousSmallRow(
                op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llLen,
                bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM, oldBlock,
                newBlock);
          }
        }
        if (!async)
          createMfenceLMOp(rewriter, loc);
        rewriter.eraseOp(op);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
      break;
    }
    case OffsetState::Unknown: {
      for (size_t llPtrIdx = 0; llPtrIdx < llPtrs.size(); ++llPtrIdx) {
        Value maskedIdx;
        if (mask) {
          auto llLenTy = llLens[0].getType();
          if (llLenTy.isInteger(32)) {
            maskedIdx = select(icmp_slt(i32_val(llPtrIdx), llLens[0]),
                               i32_val(llPtrIdx), i32_val(0));
          } else if (llLenTy.isInteger(64)) {
            maskedIdx = select(icmp_slt(i64_val(llPtrIdx), llLens[0]),
                               i64_val(llPtrIdx), i64_val(0));
          } else {
            llvm_unreachable("Unsupported Mask Int Type");
          }
        } else {
          maskedIdx = i32_val(llPtrIdx);
        }

        lmBuf = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value elemPtr = gep(ptr_ty(ctx, 0), elemTy, lmBuf, maskedIdx);
        Value srcPtr = bitcast(elemPtr, ptr_ty(ctx, 0));
        // Protect Ptr Boundary Condition
        Value dstPtr;
        if (mask) {
          auto llLenTy = llLens[0].getType();
          if (llLenTy.isInteger(32)) {
            dstPtr = select(icmp_slt(i32_val(llPtrIdx), llLens[0]),
                            llPtrs[llPtrIdx], llPtrs[0]);
          } else if (llLenTy.isInteger(64)) {
            dstPtr = select(icmp_slt(i64_val(llPtrIdx), llLens[0]),
                            llPtrs[llPtrIdx], llPtrs[0]);
          } else {
            llvm_unreachable("Unsupported Mask Int Type");
          }
        } else {
          dstPtr = llPtrs[llPtrIdx];
        }
        createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      elemBytes);
      }
      break;
    }
    default:
      llvm_unreachable("Unknown offset state");
      break;
    }
    if (!async)
      createMfenceLMOp(rewriter, loc);
    rewriter.eraseOp(op);

    return success();
  }
};

struct XPUGM2LMMaskOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GM2LMMaskOp>,
      public LoadStoreConversionBase {
  XPUGM2LMMaskOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::GM2LMMaskOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::GM2LMMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value len = op.getLen();
    Value res = op.getResult();
    int32_t tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llMask = adaptor.getMask();
    Value llLen = adaptor.getLen();
    Value llGMPtr = adaptor.getPtr();
    Value llLMPtr = adaptor.getBufPtr();
    Value resultStruct = llLMPtr;

    Type ptrTy = ptr.getType();
    Type resTy = res.getType();
    Type llvmResultStructTy = typeConverter->convertType(resTy);
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    assert(llLMPtr && "llBufPtr should not be null.");
    auto llGMPtrs = unpackLLElements(loc, llGMPtr, rewriter);
    auto llLMPtrs = unpackLLElements(loc, llLMPtr, rewriter);
    llvm::SmallVector<Value> llLens;

    Value elemBytes = i32_val(elemNbits / 8u);
    Value offsetBytes = i32_val(0);

    llvm::SmallVector<Value> llMasks;
    if (mask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }

    unsigned lenElemBit = 32;
    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (len) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      auto lenElemTy = getElementTypeOrSelf(len.getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }
    Value readBytes = mul(readLen, elemBytes);

    Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
    Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));

    int32_t fixedStride = op.getFixedStride();
    int64_t _rowLen = op.getRowLen();
    int64_t _rowStride = op.getRowStride();
    OffsetState offsetState = static_cast<OffsetState>(op.getOffsetState());
    if (offsetState == OffsetState::LocallyContinuous &&
        _rowLen % numElems == 0) {
      offsetState = OffsetState::Continuous;
      LLVM_DEBUG(
          llvm::dbgs()
          << "[OffsetState]: GM2LM Update LocallyContinuous to Continuous\n");
    }
    if (offsetState == OffsetState::Unknown) {
      /*  Small Col Size Opt Mask(14 < 16)

          Before Opt:
              T T T T T T T T
              T T T T T T F F

          After Opt:
              T T T T T T T F
              T T T T T T T F
      */
      SmallVector<bool> maskLists;
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        auto tensorRowSize =
            std::ceil(static_cast<double>(shape[0]) / 64);   // 128 / 64 = 2
        auto memColSize = shape[1];                          // 16
        unsigned rowRemainElem = memColSize - tensorColSize; // 16 - 15 = 1

        for (size_t row_idx = 0; row_idx < tensorRowSize; ++row_idx) {
          for (size_t col_idx = 0; col_idx < tensorColSize; ++col_idx) {
            maskLists.push_back(true);
          }

          for (size_t remainElem = rowRemainElem; remainElem > 0;
               --remainElem) {
            maskLists.push_back(false);
          }
        }
      }

      if (fixedStride > 0 &&
          numElems * fixedStride <= targetInfo.getXPUBufferSize()) {
        // Unknown FixedStride Vgather
        readBytes = mul(i32_val(fixedStride), readBytes);
        readBytes =
            mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
        if (!async)
          createMfenceLMOp(rewriter, loc);
      } else {
        // Unknown
        for (size_t i = 0; i < llGMPtrs.size(); ++i) {
          //  Protect Ptr Boundary Condition
          Value base = llGMPtrs[i];
          if (coreDealMultiRows) {
            base =
                len ? select(int_val(1, maskLists[i]), llGMPtrs[i], llGMPtrs[0])
                    : llGMPtrs[i];
          }
          Value dstPtr = bitcast(llLMPtrs[i], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(llGMPtrs[i], ptr_ty(ctx, 1));
          Value _readBytes =
              mask ? select(llMasks[i], elemBytes, i32_val(0)) : elemBytes;
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        _readBytes);
          if (!async)
            createMfenceLMOp(rewriter, loc);
        }
      }
      resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::Discrete) {
      // Reorder the local buffer ptrs.
      SmallVector<Value> newLmBufPtrs(llGMPtrs.size());
      Value basePtrInt = ptrtoint(i64_ty, llGMPtrs[0]);
      for (size_t idx = 0; idx < llGMPtrs.size(); ++idx) {
        Value elemPtrInt = ptrtoint(i64_ty, llGMPtrs[idx]); // convert to int
        Value offsetBytes =
            sub(elemPtrInt, basePtrInt); // get the offset(Bytes)
        Value elemPtr = gep(ptr_ty(ctx, 0), i8_ty, llLMPtrs[0], offsetBytes);
        newLmBufPtrs[idx] = elemPtr;
      }
      readBytes = mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
      createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes, readBytes);
      if (!async)
        createMfenceLMOp(rewriter, loc);

      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::DiscreteSame) {
      readBytes = elemBytes;
      readBytes = mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
      createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes, readBytes);
      if (!async)
        createMfenceLMOp(rewriter, loc);

      SmallVector<Value> newLmBufPtrs(llLMPtrs.size(), llLMPtrs[0]);
      resultStruct = packLLElements(loc, typeConverter, newLmBufPtrs, rewriter,
                                    llvmResultStructTy);
    } else if (offsetState == OffsetState::LocallyContinuous) {
      auto oldBlock = op->getBlock();
      auto newBlock = oldBlock->splitBlock(op->getNextNode());
      int64_t _elemBytes = elemNbits / 8u;
      int64_t _bufLen = static_cast<int64_t>(numElems);
      LLVM_DEBUG(llvm::dbgs() << "[GM2LM LocallyContinuous]: rowLen is "
                              << _rowLen << ", rowStride is " << _rowStride
                              << ", bufLen is " << _bufLen << "\n");
      if (_rowStride == -1) {
        lowerLocallyContinuousUnfixedStrideMask(
            op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llGMPtr, llLMPtr,
            llMask, llLen, offsetBytes, MemCpyType::GM2LM, oldBlock, newBlock);
      } else {
        if (_rowLen > _bufLen) {
          lowerLocallyContinuousUnfixedStrideMask(
              op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM,
              oldBlock, newBlock);
        } else {
          lowerLocallyContinuousSmallRowMask(
              op, loc, rewriter, _rowLen, _rowStride, llGMPtr, llLMPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::GM2LM,
              oldBlock, newBlock);
        }
      }
      if (!async)
        createMfenceLMOp(rewriter, loc);
      resultStruct = packLLElements(loc, typeConverter, llLMPtrs, rewriter,
                                    llvmResultStructTy);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
    } else if (offsetState == OffsetState::Continuous) {
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
        if (tensorColSize > 0 && tensorColSize % shape[1] == 0) {
          Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
          Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
          readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
          readBytes =
              mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
          createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        } else {
          // readBytes = mul(i32_val(tensorColSize), elemBytes);
          for (int i = 0; i < tensorRowSize; ++i) {
            Value srcPtr = bitcast(llGMPtrs[i * shape[1]], ptr_ty(ctx, 1));
            Value dstPtr = bitcast(llLMPtrs[i * shape[1]], ptr_ty(ctx, 0));
            auto _readBytes =
                mask ? select(llMasks[i * shape[1]], readBytes, i32_val(0))
                     : readBytes;
            createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                          _readBytes);
          }
        }
      } else {
        Value dstPtr = bitcast(llLMPtrs[0], ptr_ty(ctx, 0));
        Value srcPtr = bitcast(llGMPtrs[0], ptr_ty(ctx, 1));
        readBytes =
            mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
        createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[GM2LM]: offsetState is " << offsetState
                              << ", is not supported\n");
    }

    if (!async)
      createMfenceLMOp(rewriter, loc);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct XPULM2GMMaskOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::LM2GMMaskOp>,
      public LoadStoreConversionBase {

  XPULM2GMMaskOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::LM2GMMaskOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::LM2GMMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();
    Value len = op.getLen();
    int32_t offsetStateInt = op.getOffsetState();
    OffsetState offsetState = static_cast<OffsetState>(offsetStateInt);
    auto tensorColSize = op.getTensorColSize();
    bool coreDealMultiRows = tensorColSize != -1;
    offsetState = (tensorColSize == 1) ? OffsetState::Continuous : offsetState;

    bool async = op.getSyncMode() == mlir::triton::MemorySyncMode::ASYNC;

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llLen = adaptor.getLen();
    Value llBufPtr = adaptor.getBufPtr();
    assert(llBufPtr && "llBufPtr should not be null.");

    // Get elemTy and numElems
    Type ptrTy = ptr.getType();
    Type ptrElemTy = typeConverter->convertType(getElementTypeOrSelf(ptrTy));
    Type elemTy;
    if (auto ptrTensorTy = mlir::dyn_cast<RankedTensorType>(ptrTy)) {
      // Tensor
      elemTy = mlir::cast<triton::PointerType>(ptrTensorTy.getElementType())
                   .getPointeeType();
    } else {
      // Scalar
      elemTy = mlir::cast<triton::PointerType>(ptrTy).getPointeeType();
    }
    unsigned elemNbits = isa<triton::PointerType, LLVM::LLVMPointerType>(elemTy)
                             ? 64u
                             : elemTy.getIntOrFloatBitWidth();
    Value elemBytes = i32_val(elemNbits / 8u);
    unsigned numElems = getTotalElemsPerThread(ptrTy);

    // Get base, readBytes and offsetBytes
    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    llvm::SmallVector<Value> llMasks;
    llvm::SmallVector<Value> llLens;
    Value base = llPtrs[0];
    Value offsetBytes = i32_val(0);
    if (mask) {
      llMasks = unpackLLElements(loc, llMask, rewriter);
    }

    unsigned lenElemBit = 32;
    Value bufLen = i32_val(numElems);
    Value readLen = bufLen;
    if (len) {
      llLens = unpackLLElements(loc, llLen, rewriter);
      auto lenElemTy = getElementTypeOrSelf(len.getType());
      lenElemBit = lenElemTy.getIntOrFloatBitWidth();
      bufLen = int_val(lenElemBit, numElems);
      readLen = smin(smax(llLens[0], int_val(lenElemBit, 0)), bufLen);
      if (lenElemBit == 64) {
        readLen = trunc(i32_ty, readLen);
      }
    }

    Value readBytes = mul(readLen, elemBytes);
    auto lmBufPtrs = unpackLLElements(loc, llBufPtr, rewriter);
    Value lmBuf = lmBufPtrs[0];

    // Create LM2GM and mfence
    int64_t _rowLen = op.getRowLen();
    int64_t _rowStride = op.getRowStride();
    if (offsetState == OffsetState::LocallyContinuous &&
        _rowLen % numElems == 0) {
      offsetState = OffsetState::Continuous;
      LLVM_DEBUG(
          llvm::dbgs()
          << "[OffsetState]: LM2GM Update LocallyContinuous to Continuous\n");
    }
    switch (offsetState) {
    case OffsetState::Continuous: {
      if (coreDealMultiRows) {
        auto shape = cast<RankedTensorType>(ptrTy).getShape();
        int32_t tensorRowSize = std::ceil(static_cast<double>(shape[0]) / 64);
        if (tensorColSize > 0 && tensorColSize % shape[1] == 0) {
          readBytes = mul(i32_val(tensorRowSize * tensorColSize), elemBytes);
          Value srcPtr = bitcast(lmBufPtrs[0], ptr_ty(ctx, 0));
          Value dstPtr = bitcast(llPtrs[0], ptr_ty(ctx, 1));
          readBytes =
              mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
          createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                        readBytes);
        } else {
          // readBytes = mul(i32_val(tensorColSize), elemBytes);
          for (int i = 0; i < tensorRowSize; ++i) {
            Value srcPtr = bitcast(lmBufPtrs[i * shape[1]], ptr_ty(ctx, 0));
            Value dstPtr = bitcast(llPtrs[i * shape[1]], ptr_ty(ctx, 1));
            auto _readBytes =
                mask ? select(llMasks[i * shape[1]], readBytes, i32_val(0))
                     : readBytes;
            createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                          _readBytes);
          }
        }
      } else {
        Value srcPtr = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value basePtr = bitcast(base, ptr_ty(ctx, 1));
        readBytes =
            mask ? select(llMasks[0], readBytes, i32_val(0)) : readBytes;
        createLM2GMOp(rewriter, ctx, loc, srcPtr, basePtr, offsetBytes,
                      readBytes);
      }
      break;
    }
    case OffsetState::LocallyContinuous: {

      auto oldBlock = op->getBlock();
      auto newBlock = oldBlock->splitBlock(op->getNextNode());
      int64_t _elemBytes = elemNbits / 8u;
      int64_t _bufLen = static_cast<int64_t>(numElems);
      LLVM_DEBUG(llvm::dbgs() << "[LM2GM LocallyContinuous]: rowLen is "
                              << _rowLen << ", rowStride is " << _rowStride
                              << ", bufLen is " << _bufLen << "\n");

      if (_rowStride == -1) {
        lowerLocallyContinuousUnfixedStrideMask(
            op, loc, rewriter, _rowLen, _bufLen, _elemBytes, llPtr, llBufPtr,
            llMask, llLen, offsetBytes, MemCpyType::LM2GM, oldBlock, newBlock);
      } else {
        if (_rowLen > _bufLen) {
          lowerLocallyContinuousUnfixedStrideMask(
              op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM,
              oldBlock, newBlock);
        } else {
          lowerLocallyContinuousSmallRowMask(
              op, loc, rewriter, _rowLen, _rowStride, llPtr, llBufPtr, llMask,
              llLen, bufLen, elemBytes, offsetBytes, MemCpyType::LM2GM,
              oldBlock, newBlock);
        }
        if (!async)
          createMfenceLMOp(rewriter, loc);
        rewriter.eraseOp(op);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
        return success();
      }
      createMfenceLMOp(rewriter, loc);
      rewriter.create<LLVM::BrOp>(loc, ValueRange{}, newBlock);
      break;
    }
    case OffsetState::Unknown: {
      size_t ngroup = 1;
      size_t groupsize = 1;
      Value isGroupZero = icmp_eq(i32_val(0), i32_val(0));
      getLayoutInfo(value.getType(), ngroup, groupsize);
      if (ngroup * groupsize <= 64 && ngroup == 1) {
        Value coreId = mlir::LLVM::XPU::getThreadId(rewriter, loc);
        Value groupId = sdiv(coreId, i32_val(groupsize));
        isGroupZero = icmp_eq(groupId, i32_val(0));
      }
      for (size_t llPtrIdx = 0; llPtrIdx < llPtrs.size(); ++llPtrIdx) {
        Value maskedIdx = i32_val(llPtrIdx);
        Value _lmBuf = bitcast(lmBuf, ptr_ty(ctx, 0));
        Value elemPtr = gep(ptr_ty(ctx, 0), elemTy, _lmBuf, maskedIdx);
        Value srcPtr = bitcast(elemPtr, ptr_ty(ctx, 0));
        Value dstPtr = llPtrs[llPtrIdx];
        Value _mask;
        if (mask) {
          _mask = llMasks[llPtrIdx];
          if (ngroup * groupsize < 64 && ngroup == 1) {
            _mask = and_(_mask, isGroupZero);
          }
        }
        readBytes = mask ? select(_mask, elemBytes, i32_val(0)) : elemBytes;
        createLM2GMOp(rewriter, ctx, loc, srcPtr, dstPtr, offsetBytes,
                      readBytes);
        createMfenceLMOp(rewriter, loc);
      }
      break;
    }
    default: {
      llvm_unreachable("Unknown offset state");
      break;
    }
    }
    if (!async)
      createMfenceLMOp(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

struct XPUAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {

  XPUAtomicRMWOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value ptr = op.getPtr();
    Value val = op.getVal();
    Value mask = op.getMask();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llValue = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto llPtrs = unpackLLElements(loc, llPtr, rewriter);
    auto llValues = unpackLLElements(loc, llValue, rewriter);

    auto resTy = op.getType();
    Type valueElemTy = getElementTypeOrSelf(getElementTypeOrSelf(resTy));
    unsigned numElems = getTotalElemsPerThread(resTy);

    std::string funcName;
    if (valueElemTy.isF16()) {
      switch (atomicRmwAttr) {
      case RMWOp::ADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1DF16_DF16_";
        break;
      case RMWOp::FADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1DF16_DF16_";
        break;
      default:
        return failure();
      }
    } else {
      switch (atomicRmwAttr) {
      case RMWOp::ADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1ff";
        break;
      case RMWOp::FADD:
        funcName = "_ZN3xpu9atomicAddEPU3AS1ff";
        break;
      default:
        return failure();
      }
    }

    SmallVector<Value> resultVals(numElems);
    for (unsigned i = 0; i < numElems; ++i) {
      ValueRange operandRange({llPtrs[i], llValues[i]});
      Value devCall = mlir::LLVM::XPU::createDeviceCall(
          funcName, rewriter, op, valueElemTy, operandRange, loc);
      resultVals[i] = devCall;
    }

    Type structTy = this->getTypeConverter()->convertType(resTy);
    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, resultStruct);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// TLE Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower ttg::LocalAllocOp to XPU local memory allocation.
/// This is the TLE equivalent of XPUAllocaOpConversion.
///
/// SPMD model (方向 B): the memdesc shape describes the WHOLE tile, but on XPU
/// each core owns only a contiguous slice of the tile. The tile is partitioned
/// across `groupSize` (= coresPerGroup = threads-per-warp) cores, so each core
/// allocates only `ceil(tileElems / groupSize)` elements of private LM. This
/// matches make_range/local_ptr which index the buffer with per-core-local
/// indices [0, elemsPerCore).
struct XPUTLELocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>,
      public LoadStoreConversionBase {
  XPUTLELocalAllocOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy = op.getType();
    auto shape = memDescTy.getShape();
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = getTypeConverter()->convertType(elemTy);

    // Total elements in the whole tile.
    unsigned tileElems = 1;
    for (auto dim : shape)
      tileElems *= dim;

    // Per-core element count: partition the tile across cores in a group.
    // When tritonxpu-tle-core-tiling stamped a ClusterLayout, size the private
    // LM from the layout: numElems = Prod_k ceil(shape[k],
    // coresPerGroup[k]*groupsPerCluster[k]) (== getTotalElemsPerThread, and ==
    // the legacy flat cut for single-level layouts). Otherwise fall back to the
    // flat cut ceil(tileElems, groupSize).
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned elemsPerCore;
    if (auto tileLayout = op->getAttrOfType<triton::xpu::ClusterLayoutAttr>(
            "xpu.tile_layout")) {
      auto coresPerGroup = tileLayout.getCoresPerGroup();
      auto groupsPerCluster = tileLayout.getGroupsPerCluster();
      auto sizePerCore = tileLayout.getSizePerCore();
      unsigned rank = shape.size();
      if (rank == 1 && groupsPerCluster[0] > 1) {
        // LargeN result buffer: distributed over `numGroups` groups only (the
        // groupSize col-cores replicate the row), so size = ceil(M, numGroups)
        // = sizePerCore[0].
        elemsPerCore = sizePerCore[0];
      } else {
        elemsPerCore = 1;
        for (unsigned d = 0; d < rank; ++d) {
          unsigned coresAlongDim = coresPerGroup[d] * groupsPerCluster[d];
          assert(coresAlongDim > 0 &&
                 "cluster layout dim has zero cores (invalid ClusterLayout)");
          elemsPerCore *= (shape[d] + coresAlongDim - 1) / coresAlongDim;
        }
      }
    } else {
      unsigned groupSize =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      if (groupSize == 0)
        groupSize = 1;
      elemsPerCore = (tileElems + groupSize - 1) / groupSize;
    }

    auto allocNumElems = elemsPerCore;
    // XPU2 FP16: align to 32 + double space for fp16tofp32
    if (static_cast<XPUArch>(targetInfo.getXPUArch()) == XPUArch::XPU2 &&
        llvmElemTy.isF16()) {
      allocNumElems = (allocNumElems + 31) / 32 * 32;
      allocNumElems *= 2;
    }

    // 64 bytes aligned for LM
    allocNumElems = align(allocNumElems, llvmElemTy, 64);

    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    auto lmBuf = allocate(lmPtrTy, llvmElemTy, i32_val(allocNumElems));

    // For TLE, we store the base pointer as the lowered result.
    // The memdesc type will be converted to a struct containing the base ptr.
    rewriter.replaceOp(op, lmBuf);
    return success();
  }
};

/// Lower ttg::LocalLoadOp: load all elements from LM buffer into a tensor.
struct XPUTLELocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>,
      public LoadStoreConversionBase {
  XPUTLELocalLoadOpConversion(LLVMTypeConverter &converter,
                              const xpu::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    auto resTy = op.getResult().getType();
    Type llvmResultTy = typeConverter->convertType(resTy);
    Type elemTy = cast<RankedTensorType>(resTy).getElementType();
    Type llvmElemTy = typeConverter->convertType(elemTy);
    unsigned numElems = getTotalElemsPerThread(resTy);

    // src is lowered memdesc → base LM pointer
    Value lmBase = adaptor.getSrc();
    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);

    SmallVector<Value> loadedVals;
    for (unsigned i = 0; i < numElems; ++i) {
      Value elemPtr = gep(lmPtrTy, llvmElemTy, lmBase, i32_val(i));
      Value val = load(llvmElemTy, elemPtr);
      loadedVals.push_back(val);
    }

    Value resultStruct =
        packLLElements(loc, typeConverter, loadedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower ttg::LocalStoreOp: store tensor elements into LM buffer.
struct XPUTLELocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>,
      public LoadStoreConversionBase {
  XPUTLELocalStoreOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value srcVal = adaptor.getSrc();
    Value dstBuf = adaptor.getDst();

    auto srcTy = op.getSrc().getType();
    Type elemTy = cast<RankedTensorType>(srcTy).getElementType();
    Type llvmElemTy = typeConverter->convertType(elemTy);
    unsigned numElems = getTotalElemsPerThread(srcTy);

    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    auto srcVals = unpackLLElements(loc, srcVal, rewriter);

    for (unsigned i = 0; i < numElems; ++i) {
      Value elemPtr = gep(lmPtrTy, llvmElemTy, dstBuf, i32_val(i));
      store(srcVals[i], elemPtr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// One contiguous GM<->LM DMA segment for a single core.
//   gmElemOffset : element offset from the GM tile base pointer.
//   lmElemOffset : compile-time element offset into the core's private LM
//   buffer
//                  (direction-agnostic: it is the dst offset for G2L and the
//                  src offset for L2G; 0 for the single-slice case, k*innermost
//                  for the k-th row of a whole-row-multi copy).
//   validCount   : i32 number of valid elements to transfer (may be 0 to skip
//   an
//                  out-of-bounds row/slice).
struct TileDmaSeg {
  Value gmElemOffset;
  unsigned lmElemOffset;
  Value validCount;
};

// Compile-time delinearize: map a linear index to multi-dim coords over
// `sizes`, iterating dims in `order` with order[0] the fastest-varying (matches
// mlir::delinearize used by emitOffsetForClusterLayout).
static SmallVector<unsigned> delinearizeConst(unsigned linear,
                                              ArrayRef<unsigned> sizes,
                                              ArrayRef<unsigned> order) {
  unsigned rank = sizes.size();
  SmallVector<unsigned> coord(rank, 0);
  for (unsigned orderIdx = 0; orderIdx < rank && orderIdx < order.size();
       ++orderIdx) {
    unsigned d = order[orderIdx];
    unsigned dimSize = (d < rank && sizes[d]) ? sizes[d] : 1;
    coord[d] = linear % dimSize;
    linear /= dimSize;
  }
  return coord;
}

// Runtime delinearize of an i32 Value over `sizes` by `order` (order[0]
// fastest).
static SmallVector<Value> delinearizeRT(ConversionPatternRewriter &rewriter,
                                        Location loc, Value linear,
                                        ArrayRef<unsigned> sizes,
                                        ArrayRef<unsigned> order) {
  unsigned rank = sizes.size();
  SmallVector<Value> coord(rank);
  for (unsigned d = 0; d < rank; ++d)
    coord[d] = i32_val(0);
  Value cur = linear;
  for (unsigned orderIdx = 0; orderIdx < rank && orderIdx < order.size();
       ++orderIdx) {
    unsigned d = order[orderIdx];
    unsigned dimSize = (d < rank && sizes[d]) ? sizes[d] : 1;
    coord[d] = urem(cur, i32_val(dimSize));
    cur = udiv(cur, i32_val(dimSize));
  }
  return coord;
}
// Compile-time enumeration of the LM slots a core owns and the per-slot
// within-tile coordinates, derived PURELY from the ClusterLayout and buffer
// shape (no rewriter / runtime values -> unit-testable in isolation). For slot
// n: coords[n][d] = tileId[d]*shapePerCTATile[d] + elemCoord[d], i.e. the
// nano-tile origin plus the intra-nano-tile element offset -- the SAME space as
// emitOffsetForClusterLayout (Utility.cpp).
static SmallVector<SmallVector<unsigned>>
enumerateSlotCoords(triton::xpu::ClusterLayoutAttr layout,
                    ArrayRef<int64_t> bufShape) {
  unsigned rank = bufShape.size();
  auto sizePerCore = layout.getSizePerCore();
  auto coresPerGroup = layout.getCoresPerGroup();
  auto groupsPerCluster = layout.getGroupsPerCluster();
  auto order = layout.getOrder();

  SmallVector<unsigned> shapePerCTATile(rank), tilesPerDim(rank);
  unsigned totalSizePerThread = 1, numElems = 1;
  for (unsigned d = 0; d < rank; ++d) {
    shapePerCTATile[d] =
        sizePerCore[d] * coresPerGroup[d] * groupsPerCluster[d];
    tilesPerDim[d] =
        shapePerCTATile[d]
            ? (bufShape[d] + shapePerCTATile[d] - 1) / shapePerCTATile[d]
            : 1;
    unsigned coresAlongDim = coresPerGroup[d] * groupsPerCluster[d];
    assert(coresAlongDim > 0 &&
           "cluster layout dim has zero cores (invalid ClusterLayout)");
    numElems *= (bufShape[d] + coresAlongDim - 1) / coresAlongDim;
    totalSizePerThread *= sizePerCore[d];
  }

  SmallVector<SmallVector<unsigned>> coords(numElems);
  for (unsigned n = 0; n < numElems; ++n) {
    unsigned nanoId = n / totalSizePerThread;
    unsigned elemId = n % totalSizePerThread;
    SmallVector<unsigned> tileId = delinearizeConst(nanoId, tilesPerDim, order);
    SmallVector<unsigned> elemCoord =
        delinearizeConst(elemId, sizePerCore, order);
    coords[n].resize(rank);
    for (unsigned d = 0; d < rank; ++d)
      coords[n][d] = tileId[d] * shapePerCTATile[d] + elemCoord[d];
  }
  return coords;
}

// Coalesce consecutive owned slots that map to a contiguous GM run (identical
// within-tile coords in every dim except the innermost, which increments by 1)
// into single DMA segments. `threadBase64` + `coordRT` describe this core's
// runtime base; `coords` are the compile-time within-tile slot coordinates from
// enumerateSlotCoords. Each segment's lmElemOffset is its start slot n (the LM
// buffer is filled in slot-enumeration order).
static SmallVector<TileDmaSeg>
coalesceRuns(ConversionPatternRewriter &rewriter, Location loc,
             ArrayRef<SmallVector<unsigned>> coords, unsigned rank,
             Value tileOriginElem64, Value threadBase64,
             ArrayRef<Value> coordRT, ArrayRef<Value> descStrides,
             ValueRange offsets, ValueRange realShapes, bool hasRealShape) {
  SmallVector<TileDmaSeg> segs;
  Value zeroI32 = i32_val(0);
  unsigned numElems = coords.size();
  unsigned last = rank - 1;
  unsigned n = 0;
  while (n < numElems) {
    unsigned runStart = n;
    unsigned runLen = 1;
    while (n + runLen < numElems) {
      bool contiguous = true;
      for (unsigned d = 0; d < rank; ++d) {
        unsigned expect = coords[runStart][d] + (d == last ? runLen : 0);
        if (coords[n + runLen][d] != expect) {
          contiguous = false;
          break;
        }
      }
      if (!contiguous)
        break;
      ++runLen;
    }
    // GM element offset of the run start.
    Value gmRun = add(tileOriginElem64, threadBase64);
    for (unsigned d = 0; d < rank; ++d) {
      Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
      gmRun = add(gmRun, mul(i64_val((int64_t)coords[runStart][d]), stride));
    }
    // Valid element count: clamp against realShapes on the innermost dim; zero
    // the whole run if any outer coord is out of the real shape. Global coord
    // along dim d = offset[d] + coordRT[d] + coords[runStart][d].
    Value validCount = i32_val(runLen);
    if (hasRealShape) {
      Value lastOff = (last < offsets.size()) ? offsets[last] : zeroI32;
      Value gLastStart =
          add(add(lastOff, coordRT[last]), i32_val(coords[runStart][last]));
      validCount =
          smin(validCount, smax(sub(realShapes[last], gLastStart), zeroI32));
      for (unsigned d = 0; d + 1 < rank; ++d) {
        Value dOff = (d < offsets.size()) ? offsets[d] : zeroI32;
        Value gCoord = add(add(dOff, coordRT[d]), i32_val(coords[runStart][d]));
        validCount =
            select(icmp_slt(gCoord, realShapes[d]), validCount, zeroI32);
      }
    }
    segs.push_back({gmRun, runStart, validCount});
    n += runLen;
  }
  return segs;
}

// LargeN 1D output writeback (rank==1, groupsPerCluster[0] > 1 AND
// coresPerGroup[0] > 1): the reduce result [M] is distributed cyclically over
// `numGroups` groups (group grp owns rows grp, grp+numGroups, ...). All
// `groupSize` col-cores of a group hold the same row-sums after the cross-core
// reduce, so only col-core 0 of each group (idInGroup == 0) writes them back to
// avoid redundant DMAs. Slot k of the private buffer maps to global row
// grp + k*numGroups.
static SmallVector<TileDmaSeg> planLargeNOutputSegments(
    ConversionPatternRewriter &rewriter, Location loc,
    triton::xpu::ClusterLayoutAttr layout, Value tileOriginElem64,
    ArrayRef<Value> descStrides, ValueRange offsets, ValueRange realShapes,
    bool hasRealShape, Value idInGroup, Value groupId, unsigned numGroups) {
  SmallVector<TileDmaSeg> segs;
  Value zeroI32 = i32_val(0);
  unsigned rowsPerGroup = layout.getSizePerCore()[0]; // == ceil(M, numGroups)
  Value writeGate = icmp_eq(idInGroup, zeroI32);
  Value rowOffsetBase = (offsets.size() > 0) ? offsets[0] : zeroI32;
  for (unsigned k = 0; k < rowsPerGroup; ++k) {
    Value globalRow = add(groupId, i32_val(k * numGroups));
    Value gmElemOffset =
        add(tileOriginElem64, mul(sext(i64_ty, globalRow), descStrides[0]));
    Value validCount = i32_val(1);
    if (hasRealShape) {
      Value rowIdx = add(rowOffsetBase, globalRow);
      validCount = select(icmp_slt(rowIdx, realShapes[0]), validCount, zeroI32);
    }
    validCount = select(writeGate, validCount, zeroI32);
    segs.push_back({gmElemOffset, k, validCount});
  }
  return segs;
}

// Layout-driven per-core DMA planning: the ClusterLayout is the SINGLE source
// of truth for which global (row,col) each LM slot n holds, mirroring
// emitOffsetForClusterLayout (Utility.cpp) so memory / elementwise / reduce
// agree. Returns coalesced contiguous GM runs. `coreId` is the FULL physical
// core id (tid), split into idInGroup (%groupSize) + groupId. Delegates to:
//   * planLargeNOutputSegments  -- LargeN 1D cyclic result writeback;
//   * enumerateSlotCoords       -- pure compile-time owned-slot coordinates;
//   * coalesceRuns              -- merge slots into contiguous GM DMA runs.
static SmallVector<TileDmaSeg>
planTileSegmentsFromLayout(ConversionPatternRewriter &rewriter, Location loc,
                           triton::xpu::ClusterLayoutAttr layout,
                           ArrayRef<int64_t> bufShape, Value tileOriginElem64,
                           ArrayRef<Value> descStrides, ValueRange offsets,
                           ValueRange realShapes, Value coreId) {
  unsigned rank = bufShape.size();
  bool hasRealShape = (!realShapes.empty() && realShapes.size() == rank);
  auto sizePerCore = layout.getSizePerCore();
  auto coresPerGroup = layout.getCoresPerGroup();
  auto groupsPerCluster = layout.getGroupsPerCluster();
  auto order = layout.getOrder();
  unsigned groupSize = 1, numGroups = 1;
  for (unsigned d = 0; d < rank; ++d) {
    groupSize *= coresPerGroup[d];
    numGroups *= groupsPerCluster[d];
  }
  Value idInGroup = urem(coreId, i32_val(groupSize));
  Value groupId = udiv(coreId, i32_val(groupSize));

  // LargeN output special-case (see planLargeNOutputSegments). The
  // coresPerGroup[0] > 1 guard distinguishes the g > 1 (LargeN) case from the
  // g == 1 (RowTiled) 1D output whose layout is cpg=[1]/gpc=[coreNum]: there
  // groupsPerCluster[0] > 1 but coresPerGroup[0] == 1, so it falls through to
  // the general BLOCK branch below (which is what RowTiled needs; cyclic would
  // be wrong for it).
  if (rank == 1 && groupsPerCluster[0] > 1 && coresPerGroup[0] > 1)
    return planLargeNOutputSegments(
        rewriter, loc, layout, tileOriginElem64, descStrides, offsets,
        realShapes, hasRealShape, idInGroup, groupId, numGroups);

  // ---- General branch: RowTiled / LargeN / 1D / default-1D. ----
  // Runtime per-core coords: coreCoordInGroup[d] = intra-group core coord
  // (idInGroup over coresPerGroup), groupCoord[d] = group coord (groupId over
  // groupsPerCluster), both delinearized by `order`. The per-core thread base
  // coordinate along dim d is
  //   coordRT[d] = groupCoord[d]*(sizePerCore[d]*coresPerGroup[d])
  //                + coreCoordInGroup[d]*sizePerCore[d].
  SmallVector<Value> coreCoordInGroup =
      delinearizeRT(rewriter, loc, idInGroup, coresPerGroup, order);
  SmallVector<Value> groupCoord =
      delinearizeRT(rewriter, loc, groupId, groupsPerCluster, order);
  SmallVector<Value> coordRT(rank);
  Value threadBase64 = i64_val(0);
  for (unsigned d = 0; d < rank; ++d) {
    coordRT[d] =
        add(mul(groupCoord[d], i32_val(sizePerCore[d] * coresPerGroup[d])),
            mul(coreCoordInGroup[d], i32_val(sizePerCore[d])));
    Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
    threadBase64 = add(threadBase64, mul(sext(i64_ty, coordRT[d]), stride));
  }

  // Enumerate owned slots (pure compile-time) then coalesce into GM runs.
  SmallVector<SmallVector<unsigned>> coords =
      enumerateSlotCoords(layout, bufShape);
  if (coords.empty())
    return {};
  return coalesceRuns(rewriter, loc, coords, rank, tileOriginElem64,
                      threadBase64, coordRT, descStrides, offsets, realShapes,
                      hasRealShape);
}

// ---- Legacy flat row-major cut (used when NO xpu.tile_layout is stamped).
// ---- Core c owns the flattened element range [c*epc, (c+1)*epc) of the tile.
// This is the pre-CoreTiling behavior, kept verbatim for TLE kernels that do
// not run tritonxpu-tle-core-tiling.
static SmallVector<TileDmaSeg>
planTileSegmentsLegacy(ConversionPatternRewriter &rewriter, Location loc,
                       ArrayRef<int64_t> bufShape, Value tileOriginElem64,
                       ArrayRef<Value> descStrides, ValueRange offsets,
                       ValueRange realShapes, ArrayRef<Value> coreCoord,
                       Value coreBase, Value gmElemOffset,
                       unsigned elemsPerCore, unsigned tileElems) {
  unsigned rank = bufShape.size();
  bool hasRealShape = (!realShapes.empty() && realShapes.size() == rank);
  Value zeroI32 = i32_val(0);
  SmallVector<TileDmaSeg> segs;

  unsigned innermost = (rank >= 1) ? bufShape[rank - 1] : 0;
  // Does each core own MULTIPLE WHOLE innermost rows? Those rows are
  // non-contiguous in GM once realN != innermost or a col offset is applied, so
  // they need one DMA each; otherwise a single slice suffices.
  bool wholeRowMulti =
      (rank == 2 && innermost > 0 && elemsPerCore % innermost == 0 &&
       elemsPerCore / innermost > 1);
  if (wholeRowMulti) {
    unsigned rowsPerCore = elemsPerCore / innermost;
    Value colOff = (offsets.size() > (rank - 1)) ? offsets[rank - 1] : zeroI32;
    Value validCols = i32_val(innermost);
    if (hasRealShape) {
      Value remCols = smax(sub(realShapes[rank - 1], colOff), zeroI32);
      validCols = smin(validCols, remCols);
    }
    Value rowOffsetBase = (offsets.size() > 0) ? offsets[0] : zeroI32;
    for (unsigned k = 0; k < rowsPerCore; ++k) {
      Value rowCoord = add(coreCoord[0], i32_val(k)); // buffer row index
      Value gmElemOffRow =
          add(tileOriginElem64, mul(sext(i64_ty, rowCoord), descStrides[0]));
      Value validCountRow = validCols;
      if (hasRealShape) {
        Value globalRow = add(rowOffsetBase, rowCoord);
        Value inBound = icmp_slt(globalRow, realShapes[0]);
        validCountRow = select(inBound, validCountRow, zeroI32);
      }
      segs.push_back({gmElemOffRow, k * innermost, validCountRow});
    }
  } else {
    // Single per-core slice, staying within one innermost row.
    Value remainElems = smax(sub(i32_val(tileElems), coreBase), zeroI32);
    Value validCount = smin(remainElems, i32_val(elemsPerCore));
    if (hasRealShape) {
      unsigned last = rank - 1;
      Value colOff = (last < offsets.size()) ? offsets[last] : zeroI32;
      Value gColStart = add(colOff, coreCoord[last]);
      Value remCols = smax(sub(realShapes[last], gColStart), zeroI32);
      validCount = smin(validCount, remCols);
      for (unsigned d = 0; d + 1 < rank; ++d) {
        Value dOff = (d < offsets.size()) ? offsets[d] : zeroI32;
        Value gCoord = add(dOff, coreCoord[d]);
        Value inBound = icmp_slt(gCoord, realShapes[d]);
        validCount = select(inBound, validCount, zeroI32);
      }
    }
    segs.push_back({gmElemOffset, 0, validCount});
  }
  return segs;
}

/// Lower triton_xpu.tle_copy_g2l to GM2LM DMA instruction.
struct XPUTLECopyG2LOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLECopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  XPUTLECopyG2LOpConversion(LLVMTypeConverter &converter,
                            const xpu::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLECopyGlobalToLocalOp>(converter,
                                                                    benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLECopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Get the descriptor (TensorDescType is lowered to ptr<1> = GM base
    // pointer)
    Value desc = adaptor.getDesc();
    // Get dst buffer (LM base pointer)
    Value dstBuf = adaptor.getDstBuffer();
    // Get offsets
    auto offsets = adaptor.getOffsets();

    // desc is already a GM pointer (ptr<1>), no extraction needed.
    Value gmBasePtr = desc;

    // Strides are passed directly as op operands (desc.strides from Python).
    SmallVector<Value> descStrides;
    for (auto s : adaptor.getStrides())
      descStrides.push_back(s);
    // Fallback for descriptors without explicit strides: row-major contiguous.
    if (descStrides.empty()) {
      auto bufShape2 =
          cast<triton::gpu::MemDescType>(op.getDstBuffer().getType())
              .getShape();
      unsigned rank = bufShape2.size();
      unsigned s0 = 1;
      for (unsigned j = 1; j < rank; ++j)
        s0 *= bufShape2[j];
      descStrides.push_back(i64_val(s0));
      if (rank > 1)
        descStrides.push_back(i64_val(1));
    }

    // Get the memdesc type to know element type and shape
    auto memDescTy = op.getDstBuffer().getType();
    auto bufShape = cast<triton::gpu::MemDescType>(memDescTy).getShape();
    Type elemTy = cast<triton::gpu::MemDescType>(memDescTy).getElementType();
    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;

    // Support arbitrary rank tensors. The memdesc shape is the WHOLE tile.
    unsigned rank = bufShape.size();
    unsigned tileElems = 1;
    for (unsigned d = 0; d < rank; ++d)
      tileElems *= bufShape[d];

    // --- SPMD per-core partition (方向 B) ---
    // The tile is split contiguously across `groupSize` cores. Core `c` owns
    // the flattened element range [c*elemsPerCore, (c+1)*elemsPerCore) and
    // copies it into its private LM buffer starting at local index 0.
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned groupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (groupSize == 0)
      groupSize = 1;
    unsigned elemsPerCore = (tileElems + groupSize - 1) / groupSize;

    // Tile-origin element offset in GM: sum(offset[i] * stride[i]).
    Value tileOriginElem64 = i64_val(0);
    for (unsigned i = 0; i < offsets.size() && i < descStrides.size(); ++i) {
      Value off_64 = sext(i64_ty, offsets[i]);
      tileOriginElem64 = add(tileOriginElem64, mul(off_64, descStrides[i]));
    }

    // Per-core flattened start index within the tile.
    Value coreId = tid_val();
    Value idInsideGroup = srem(coreId, i32_val(groupSize));
    Value coreBase = mul(idInsideGroup, i32_val(elemsPerCore));

    // Decompose coreBase into multi-dim coords (row-major over bufShape) and
    // compute the GM element offset of the core's first element.
    Value gmElemOffset = tileOriginElem64;
    Value rem = coreBase;
    SmallVector<Value> coreCoord(rank);
    for (unsigned d = 0; d < rank; ++d) {
      unsigned innerProd = 1;
      for (unsigned j = d + 1; j < rank; ++j)
        innerProd *= bufShape[j];
      Value coord = (innerProd > 1) ? udiv(rem, i32_val(innerProd)) : rem;
      if (d + 1 < rank)
        rem = urem(rem, i32_val(innerProd));
      coreCoord[d] = coord;
      Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
      gmElemOffset = add(gmElemOffset, mul(sext(i64_ty, coord), stride));
    }
    Value zeroI32 = i32_val(0);
    Value dstPtr = bitcast(dstBuf, ptr_ty(ctx, 0));
    auto shapesG2L = adaptor.getShapes();

    // Plan the per-core DMA segments. When tritonxpu-tle-core-tiling stamped a
    // ClusterLayout on this op ("xpu.tile_layout"), the partition is fully
    // layout-driven; otherwise fall back to the legacy flat row-major cut.
    auto tileLayout =
        op->getAttrOfType<triton::xpu::ClusterLayoutAttr>("xpu.tile_layout");
    SmallVector<TileDmaSeg> segs =
        tileLayout
            ? planTileSegmentsFromLayout(rewriter, loc, tileLayout, bufShape,
                                         tileOriginElem64, descStrides, offsets,
                                         shapesG2L, coreId)
            : planTileSegmentsLegacy(rewriter, loc, bufShape, tileOriginElem64,
                                     descStrides, offsets, shapesG2L, coreCoord,
                                     coreBase, gmElemOffset, elemsPerCore,
                                     tileElems);

    for (auto &seg : segs) {
      Value gmByteOff = mul(seg.gmElemOffset, i64_val(elemBytes));
      Value gmAddr = gep(ptr_ty(ctx, 1), i8_ty, gmBasePtr, gmByteOff);
      Value srcPtr = bitcast(gmAddr, ptr_ty(ctx, 1));
      Value dstPtrSeg =
          seg.lmElemOffset == 0
              ? dstPtr
              : gep(ptr_ty(ctx, 0), elemTy, dstPtr, i32_val(seg.lmElemOffset));
      Value copyBytes = mul(seg.validCount, i32_val(elemBytes));
      createGM2LMOp(rewriter, ctx, loc, srcPtr, dstPtrSeg, zeroI32, copyBytes);
    }

    bool isSync = op.getIsSync();
    if (isSync)
      createMfenceOp(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_copy_l2g to LM2GM DMA instruction.
struct XPUTLECopyL2GOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLECopyLocalToGlobalOp>,
      public LoadStoreConversionBase {
  XPUTLECopyL2GOpConversion(LLVMTypeConverter &converter,
                            const xpu::TargetInfo &targetInfo,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLECopyLocalToGlobalOp>(converter,
                                                                    benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLECopyLocalToGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value desc = adaptor.getDesc();
    Value srcBuf = adaptor.getSrcBuffer();
    auto offsets = adaptor.getOffsets();

    // desc is a GM pointer (ptr<1>)
    Value gmBasePtr = desc;

    // Strides are passed directly as op operands (desc.strides from Python).
    SmallVector<Value> descStrides;
    for (auto s : adaptor.getStrides())
      descStrides.push_back(s);
    // Fallback for descriptors without explicit strides: row-major contiguous.
    if (descStrides.empty()) {
      auto bufShape2 =
          cast<triton::gpu::MemDescType>(op.getSrcBuffer().getType())
              .getShape();
      unsigned rank = bufShape2.size();
      unsigned s0 = 1;
      for (unsigned j = 1; j < rank; ++j)
        s0 *= bufShape2[j];
      descStrides.push_back(i64_val(s0));
      if (rank > 1)
        descStrides.push_back(i64_val(1));
    }

    // Get buffer shape/type info
    auto memDescTy = op.getSrcBuffer().getType();
    auto bufShape = cast<triton::gpu::MemDescType>(memDescTy).getShape();
    Type elemTy = cast<triton::gpu::MemDescType>(memDescTy).getElementType();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

    // Support arbitrary rank tensors. The memdesc shape is the WHOLE tile.
    unsigned rank = bufShape.size();
    unsigned tileElems = 1;
    for (unsigned d = 0; d < rank; ++d)
      tileElems *= bufShape[d];

    // --- SPMD per-core partition (方向 B) ---
    // Core `c` writes back only the flattened element range
    // [c*elemsPerCore, (c+1)*elemsPerCore) from its private LM buffer to GM.
    auto mod = op->getParentOfType<ModuleOp>();
    unsigned groupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (groupSize == 0)
      groupSize = 1;
    unsigned elemsPerCore = (tileElems + groupSize - 1) / groupSize;

    // Tile-origin element offset in GM: sum(offset[i] * stride[i]).
    Value tileOriginElem64 = i64_val(0);
    for (unsigned i = 0; i < offsets.size() && i < descStrides.size(); ++i) {
      Value off_64 = sext(i64_ty, offsets[i]);
      tileOriginElem64 = add(tileOriginElem64, mul(off_64, descStrides[i]));
    }

    // Per-core flattened start index within the tile.
    Value coreId = tid_val();
    Value idInsideGroup = srem(coreId, i32_val(groupSize));
    Value coreBase = mul(idInsideGroup, i32_val(elemsPerCore));

    // Decompose coreBase into multi-dim coords (row-major over bufShape) and
    // compute the GM element offset of the core's first element.
    Value gmElemOffset = tileOriginElem64;
    Value rem = coreBase;
    SmallVector<Value> coreCoord(rank);
    for (unsigned d = 0; d < rank; ++d) {
      unsigned innerProd = 1;
      for (unsigned j = d + 1; j < rank; ++j)
        innerProd *= bufShape[j];
      Value coord = (innerProd > 1) ? udiv(rem, i32_val(innerProd)) : rem;
      if (d + 1 < rank)
        rem = urem(rem, i32_val(innerProd));
      coreCoord[d] = coord;
      Value stride = (d < descStrides.size()) ? descStrides[d] : i64_val(1);
      gmElemOffset = add(gmElemOffset, mul(sext(i64_ty, coord), stride));
    }
    Value zeroI32 = i32_val(0);
    Value srcLmPtr = bitcast(srcBuf, ptr_ty(ctx, 0));
    auto shapesL2G = adaptor.getShapes();

    // Plan the per-core DMA using the SAME layout-driven dispatcher as copy_g2l
    // (segments are direction-agnostic: {gmElemOffset, lmElemOffset,
    // validCount}). When absent, the legacy flat cut applies (the reduce OUTPUT
    // buffer is 1D [XBLOCK], so the layout path enumerates one contiguous
    // per-core slice too).
    auto tileLayout =
        op->getAttrOfType<triton::xpu::ClusterLayoutAttr>("xpu.tile_layout");
    SmallVector<TileDmaSeg> segs =
        tileLayout
            ? planTileSegmentsFromLayout(rewriter, loc, tileLayout, bufShape,
                                         tileOriginElem64, descStrides, offsets,
                                         shapesL2G, coreId)
            : planTileSegmentsLegacy(rewriter, loc, bufShape, tileOriginElem64,
                                     descStrides, offsets, shapesL2G, coreCoord,
                                     coreBase, gmElemOffset, elemsPerCore,
                                     tileElems);

    // Ensure preceding CPU stores to LM are visible to the DMA engine.
    createMfenceOp(rewriter, loc);
    for (auto &seg : segs) {
      Value gmByteOff = mul(seg.gmElemOffset, i64_val(elemBytes));
      Value gmAddr = gep(ptr_ty(ctx, 1), i8_ty, gmBasePtr, gmByteOff);
      Value dstPtr = bitcast(gmAddr, ptr_ty(ctx, 1));
      Value srcPtrSeg = seg.lmElemOffset == 0
                            ? srcLmPtr
                            : gep(ptr_ty(ctx, 0), elemTy, srcLmPtr,
                                  i32_val(seg.lmElemOffset));
      Value copyBytesVal = mul(seg.validCount, i32_val(elemBytes));
      createLM2GMOp(rewriter, ctx, loc, srcPtrSeg, dstPtr, zeroI32,
                    copyBytesVal);
    }
    createMfenceOp(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_normcopy_g2l: element-wise (permuted) gather from a GM
/// pointer tensor into a local memory buffer. Element i of this core's lane set
/// is copied to local slot i, matching the tle_local_ptr layout.
struct XPUTLENormCopyG2LOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLENormCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  XPUTLENormCopyG2LOpConversion(LLVMTypeConverter &converter,
                                const xpu::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLENormCopyGlobalToLocalOp>(
            converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLENormCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy =
        cast<triton::gpu::MemDescType>(op.getDstBuffer().getType());
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = getTypeConverter()->convertType(elemTy);
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

    // Per-core GM pointer lanes (same distribution as tle_local_ptr).
    auto gmPtrs = unpackLLElements(loc, adaptor.getSrcPtrs(), rewriter);
    Value lmBase = bitcast(adaptor.getDstBuffer(), ptr_ty(ctx, 0));
    Value zeroI32 = i32_val(0);
    Value szI32 = i32_val(elemBytes);

    for (unsigned i = 0; i < gmPtrs.size(); ++i) {
      Value gmPtr = bitcast(gmPtrs[i], ptr_ty(ctx, 1));
      Value lmSlot = gep(ptr_ty(ctx, 0), llvmElemTy, lmBase, i32_val(i));
      createGM2LMOp(rewriter, ctx, loc, gmPtr, lmSlot, zeroI32, szI32);
    }
    createMfenceOp(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_normcopy_l2g: element-wise (permuted) scatter from a
/// local memory buffer to a GM pointer tensor. Scatter counterpart of the
/// gather above.
struct XPUTLENormCopyL2GOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLENormCopyLocalToGlobalOp>,
      public LoadStoreConversionBase {
  XPUTLENormCopyL2GOpConversion(LLVMTypeConverter &converter,
                                const xpu::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLENormCopyLocalToGlobalOp>(
            converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLENormCopyLocalToGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto memDescTy =
        cast<triton::gpu::MemDescType>(op.getSrcBuffer().getType());
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = getTypeConverter()->convertType(elemTy);
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;

    auto gmPtrs = unpackLLElements(loc, adaptor.getDstPtrs(), rewriter);
    Value lmBase = bitcast(adaptor.getSrcBuffer(), ptr_ty(ctx, 0));
    Value zeroI32 = i32_val(0);
    Value szI32 = i32_val(elemBytes);

    // Ensure preceding CPU stores to LM are visible to the DMA engine.
    createMfenceOp(rewriter, loc);
    for (unsigned i = 0; i < gmPtrs.size(); ++i) {
      Value lmSlot = gep(ptr_ty(ctx, 0), llvmElemTy, lmBase, i32_val(i));
      Value gmPtr = bitcast(gmPtrs[i], ptr_ty(ctx, 1));
      createLM2GMOp(rewriter, ctx, loc, lmSlot, gmPtr, zeroI32, szI32);
    }
    createMfenceOp(rewriter, loc);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_local_ptr: compute LM pointers from buffer + indices.
struct XPUTLELocalPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLELocalPtrOp>,
      public LoadStoreConversionBase {
  XPUTLELocalPtrOpConversion(LLVMTypeConverter &converter,
                             const xpu::TargetInfo &targetInfo,
                             ModuleAxisInfoAnalysis &axisAnalysisPass,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLELocalPtrOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLELocalPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // Get buffer base pointer (lowered from memdesc)
    Value lmBase = adaptor.getBuffer();

    // Get the result type to determine shape.
    // On XPU, local memory lives in addrspace 0 (flat pointer space).
    // Even though Triton represents shared-memory pointers as addrspace 3,
    // the XPU backend keeps LM allocations in addrspace 0 and all
    // loads/stores go through addrspace 0 pointers.
    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    unsigned addrSpace = 0;
    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, addrSpace);

    Type llvmResultTy = typeConverter->convertType(resTy);

    // Get buffer element type from the memdesc
    auto memDescTy = cast<triton::gpu::MemDescType>(op.getBuffer().getType());
    auto bufShape = memDescTy.getShape();
    Type elemTy = memDescTy.getElementType();
    Type llvmElemTy = typeConverter->convertType(elemTy);

    // Get index operands
    auto indices = adaptor.getIndices();
    unsigned rank = bufShape.size();

    // Calculate the number of result pointers (per-thread elements).
    // Since the result type is an unencoded pointer tensor, use the LLVM
    // struct type from the type converter (which computes proper numElems).
    auto llvmResTy = cast<LLVM::LLVMStructType>(llvmResultTy);
    unsigned numElems = llvmResTy.getBody().size();

    // --- SPMD per-core partition (方向 B) ---
    // The LM buffer is per-core (elemsPerCore elements). make_range distributes
    // the tile so this core's i-th element is global tile position
    // coreBase + i, and CopyG2L placed that element at LM-local slot i.
    // Therefore the i-th pointer is simply lmBase + i (local index). The raw
    // global index operands are not needed here.
    (void)indices;
    (void)rank;
    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    SmallVector<Value> resultPtrs;
    for (unsigned i = 0; i < numElems; ++i) {
      Value ptr = gep(basePtrTy, llvmElemTy, lmBase, i32_val(i));
      resultPtrs.push_back(ptr);
    }

    Value resultStruct =
        packLLElements(loc, typeConverter, resultPtrs, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton::LoadOp in TLE path: simple element-wise load from LM pointers.
/// This handles the pattern: tle_local_ptr → tl.load (LM pointer load).
struct XPUTLETritonLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::LoadOp>,
      public LoadStoreConversionBase {
  XPUTLETritonLoadOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Type resTy = op.getResult().getType();
    Type llvmResultTy = typeConverter->convertType(resTy);
    Type elemTy = getElementTypeOrSelf(resTy);
    Type llvmElemTy = typeConverter->convertType(elemTy);

    auto lmPtrTy = LLVM::LLVMPointerType::get(ctx, 0);

    // Unpack pointer values (each is an LM pointer from tle_local_ptr)
    auto llPtrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    unsigned numElems = llPtrs.size();

    SmallVector<Value> loadedVals;
    for (unsigned i = 0; i < numElems; ++i) {
      Value val = load(llvmElemTy, llPtrs[i]);
      loadedVals.push_back(val);
    }

    Value resultStruct =
        packLLElements(loc, typeConverter, loadedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton::StoreOp in TLE path: simple element-wise store to LM pointers.
struct XPUTLETritonStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  XPUTLETritonStoreOpConversion(LLVMTypeConverter &converter,
                                const xpu::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto llPtrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto llVals = unpackLLElements(loc, adaptor.getValue(), rewriter);

    for (unsigned i = 0; i < llPtrs.size(); ++i) {
      store(llVals[i], llPtrs[i]);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower triton_xpu.tle_vload: vectorized load from a per-core LM buffer.
/// The result tensor element type is a VectorType<W x elem>; each struct slot
/// i loads W contiguous elements from LM[i*W : (i+1)*W] as one <W x elem>.
struct XPUTLEVLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLEVLoadOp>,
      public LoadStoreConversionBase {
  XPUTLEVLoadOpConversion(LLVMTypeConverter &converter,
                          const xpu::TargetInfo &targetInfo,
                          ModuleAxisInfoAnalysis &axisAnalysisPass,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLEVLoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLEVLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = adaptor.getBuffer();

    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    Type vecElemTy = resTensorTy.getElementType(); // vector<W x elem>
    Type llvmVecTy = typeConverter->convertType(vecElemTy);
    Type llvmResultTy = typeConverter->convertType(resTy);
    auto llvmStructTy = cast<LLVM::LLVMStructType>(llvmResultTy);
    unsigned numVecs = llvmStructTy.getBody().size();

    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    SmallVector<Value> loadedVals;
    for (unsigned i = 0; i < numVecs; ++i) {
      Value ptr = gep(basePtrTy, llvmVecTy, lmBase, i32_val(i));
      Value v = load(llvmVecTy, ptr);
      loadedVals.push_back(v);
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, loadedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton_xpu.tle_vstore: vectorized store to a per-core LM buffer.
struct XPUTLEVStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::TLEVStoreOp>,
      public LoadStoreConversionBase {
  XPUTLEVStoreOpConversion(LLVMTypeConverter &converter,
                           const xpu::TargetInfo &targetInfo,
                           ModuleAxisInfoAnalysis &axisAnalysisPass,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::TLEVStoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::TLEVStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = adaptor.getBuffer();
    auto valTensorTy = cast<RankedTensorType>(op.getValue().getType());
    Type vecElemTy = valTensorTy.getElementType(); // vector<W x elem>
    Type llvmVecTy = typeConverter->convertType(vecElemTy);

    auto llVals = unpackLLElements(loc, adaptor.getValue(), rewriter);
    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);
    for (unsigned i = 0; i < llVals.size(); ++i) {
      Value ptr = gep(basePtrTy, llvmVecTy, lmBase, i32_val(i));
      store(llVals[i], ptr);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Shared helper for the vector<->scalar boundary ops: recover the LM base
/// pointer from the bufPtr operand. XPUAllocaOpConversion materializes an
/// alloca as a struct of per-element pointers, so the buffer base is the
/// first element of that struct.
static Value getBoundaryLMBase(Location loc, Value bufPtr,
                               ConversionPatternRewriter &rewriter) {
  if (!bufPtr)
    return Value();
  if (isa<LLVM::LLVMStructType>(bufPtr.getType()))
    return unpackLLElements(loc, bufPtr, rewriter)[0];
  return bufPtr;
}

/// Lower triton_xpu.pack: scalar tensor -> vector tensor, through LM.
/// The scalar elements are stored contiguously into the buffer and read back
/// as whole vectors, one vector load per result slot. Deliberately no
/// per-lane insertelement chain: the point of this op is to keep the
/// reinterpretation at one memory op per slot.
struct XPUPackOpConversion : public ConvertOpToLLVMPattern<triton::xpu::PackOp>,
                             public LoadStoreConversionBase {
  XPUPackOpConversion(LLVMTypeConverter &converter,
                      const xpu::TargetInfo &targetInfo,
                      ModuleAxisInfoAnalysis &axisAnalysisPass,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::PackOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = getBoundaryLMBase(loc, adaptor.getBufPtr(), rewriter);
    if (!lmBase)
      return op.emitError("triton_xpu.pack reached lowering without a bufPtr; "
                          "tritonxpu-alloca must attach one");

    auto srcTensorTy = cast<RankedTensorType>(op.getSrc().getType());
    Type llvmScalarTy =
        typeConverter->convertType(srcTensorTy.getElementType());

    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    Type llvmVecTy =
        typeConverter->convertType(resTensorTy.getElementType()); // <W x elem>
    Type llvmResultTy = typeConverter->convertType(resTy);
    unsigned numVecs =
        cast<LLVM::LLVMStructType>(llvmResultTy).getBody().size();

    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);

    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    for (unsigned i = 0; i < srcVals.size(); ++i) {
      Value ptr = gep(basePtrTy, llvmScalarTy, lmBase, i32_val(i));
      store(srcVals[i], ptr);
    }
    // The scalar stores and the vector loads below hit the same LM bytes from
    // two different pipes, and the hardware does not order them: without this
    // fence the widest load picks up stale bytes for whichever elements were
    // stored last. Measured on `add` (8 Mi f32) -- exactly the tail of each
    // 64-scalar group came back wrong (offsets 61..63 and 125..127 of the
    // 128-element per-core slice), nondeterministically, ~2% of elements.
    // Mask 1 is LM only; the GM bit (4) that createMfenceOp uses is not needed
    // here and would make every boundary pay for a DMA fence it never issues.
    createMfenceLMOp(rewriter, loc);

    SmallVector<Value> packedVals;
    for (unsigned i = 0; i < numVecs; ++i) {
      Value ptr = gep(basePtrTy, llvmVecTy, lmBase, i32_val(i));
      packedVals.push_back(load(llvmVecTy, ptr));
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, packedVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

/// Lower triton_xpu.unpack: vector tensor -> scalar tensor, through LM.
/// Inverse of XPUPackOpConversion: one vector store per source slot, then a
/// scalar load per result element.
struct XPUUnpackOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::UnpackOp>,
      public LoadStoreConversionBase {
  XPUUnpackOpConversion(LLVMTypeConverter &converter,
                        const xpu::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::UnpackOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::xpu::UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    Value lmBase = getBoundaryLMBase(loc, adaptor.getBufPtr(), rewriter);
    if (!lmBase)
      return op.emitError("triton_xpu.unpack reached lowering without a "
                          "bufPtr; tritonxpu-alloca must attach one");

    auto srcTensorTy = cast<RankedTensorType>(op.getSrc().getType());
    Type llvmVecTy =
        typeConverter->convertType(srcTensorTy.getElementType()); // <W x elem>

    auto resTy = op.getResult().getType();
    auto resTensorTy = cast<RankedTensorType>(resTy);
    Type llvmScalarTy =
        typeConverter->convertType(resTensorTy.getElementType());
    Type llvmResultTy = typeConverter->convertType(resTy);
    unsigned numElems =
        cast<LLVM::LLVMStructType>(llvmResultTy).getBody().size();

    auto basePtrTy = LLVM::LLVMPointerType::get(ctx, 0);

    // Volatile in both directions, and not for ordering -- the fence below does
    // that. Without it LLVM forwards the vector store into the scalar loads
    // (the LM buffer SROAs away entirely) and re-expresses the reinterpretation
    // as `bitcast <16 x i32> to <64 x i8>` + 16 `extractelement`s. The XPU3
    // backend lowers that by spilling the vector register to the *stack*
    // (`vstore_mask64.mz vr1{mr1}, -704(r30)`) and reading bytes back out, and
    // that faults on device: `truncint` at any size died with
    // cudaErrorLaunchFailure until these accesses became volatile. The LM round
    // trip is the semantics of this op, not an implementation detail -- it is
    // also what P4's `N + ceil(N/W) + 1` price counts, so letting it vanish
    // would make the boundary unmeasurable even where it does not fault.
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    for (unsigned i = 0; i < srcVals.size(); ++i) {
      Value ptr = gep(basePtrTy, llvmVecTy, lmBase, i32_val(i));
      store(srcVals[i], ptr, /*alignment=*/0, /*isVolatile=*/true);
    }
    // Same hazard as in the pack direction, stores and loads swapped.
    createMfenceLMOp(rewriter, loc);

    SmallVector<Value> scalarVals;
    for (unsigned i = 0; i < numElems; ++i) {
      Value ptr = gep(basePtrTy, llvmScalarTy, lmBase, i32_val(i));
      scalarVals.push_back(
          load(llvmScalarTy, ptr, /*alignment=*/0, /*isVolatile=*/true));
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, scalarVals, rewriter, llvmResultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns
      .add<XPULoadOpConversion, XPUStoreOpConversion, XPUAllocaOpConversion,
           XPULoadScalarIndexedOpConversion, XPUStageSMOpConversion,
           XPUGM2LMMaskOpConversion, XPULM2GMMaskOpConversion,
           XPUAtomicRMWOpConversion, XPUGM2LMOpConversion, XPULM2GMOpConversion,
           // TLE patterns
           XPUTLELocalAllocOpConversion, XPUTLECopyG2LOpConversion,
           XPUTLECopyL2GOpConversion, XPUTLENormCopyG2LOpConversion,
           XPUTLENormCopyL2GOpConversion, XPUTLELocalPtrOpConversion,
           XPUTLETritonLoadOpConversion, XPUTLETritonStoreOpConversion,
           XPUTLEVLoadOpConversion, XPUTLEVStoreOpConversion,
           // vector<->scalar boundary
           XPUPackOpConversion, XPUUnpackOpConversion>(
          typeConverter, targetInfo, axisInfoAnalysis, benefit);
}
