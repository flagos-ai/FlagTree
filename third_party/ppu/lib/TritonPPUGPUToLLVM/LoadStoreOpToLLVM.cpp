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
#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "TritonPPUGPUToLLVM/TIXAsmFormat.h"

#include "Dialect/TritonPPUGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

#include <cassert>

using namespace mlir;
using namespace mlir::triton;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir::triton::ppu;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::NVMMASharedEncodingAttr;

// Toggle this to work around Cooperative Grid Launch ld.acquire optimized path
static constexpr bool disableLDAcquireLowering = false;

namespace {

Value maybeAnd(RewriterBase &rewriter, Location loc, Value a, Value b) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  if (a && b) {
    return tb.and_(a, b);
  }
  return a ? a : b;
}

// Check whether a pointer Value points to shared memory (addrspace 3).
// TLE's local_pointers lowering produces pointers in this space; using
// global-memory instructions on them crashes PPU hardware.
bool isSharedMemoryPointer(Value ptr) {
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ptr.getType()))
    return ptrTy.getAddressSpace() == 3;
  return false;
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask. The predicate may be null to indicate no
// predication is required.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const ppu::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred;
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = maybeAnd(rewriter, loc, pred, dimPred);
    }
  }
  return pred;
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

std::string getRegisterSizeCode(int size, bool is_float) {
  switch (size) {
  case 1:
    return "b";
  case 16:
    return "h";
  case 32:
    return is_float ? "f" : "r";
  case 64:
    return is_float ? "d" : "l";
  case 128:
    return "q";
  default:
    llvm_unreachable("Unsupported register size");
  }
}

Value createCachePolicy(triton::EvictionPolicy opEvict,
                        ConversionPatternRewriter &rewriter, Location loc,
                        int computeCapability) {
  // Emit createpolicy.fractional.L2::policy.b64 xx 1.0
  TIXBuilder tixBuilder;
  const bool hasL2EvictPolicy =
      opEvict == triton::EvictionPolicy::EVICT_FIRST ||
      opEvict == triton::EvictionPolicy::EVICT_LAST;
  Value policyRet;

  const bool hardwareSupport = computeCapability >= 80;

  if (hasL2EvictPolicy && hardwareSupport) {
    auto &policy =
        tixBuilder.create("ppu.createpolicy.fractional")
            ->o("L2::evict_first",
                opEvict == triton::EvictionPolicy::EVICT_FIRST)
            .o("L2::evict_last", opEvict == triton::EvictionPolicy::EVICT_LAST)
            .b(64);

    const std::string writeConstraint = "=l";
    // prepare asm operands
    auto *dstOpr = tixBuilder.newOperand(writeConstraint, /*init=*/true);
    std::string fractionStr = "1.0";
    auto *fractionOpr = tixBuilder.newConstantOperand(fractionStr);
    policy(dstOpr, fractionOpr);

    Type policyRetTy = rewriter.getI64Type();
    policyRet = tixBuilder.launch(rewriter, loc, policyRetTy);
  }

  return policyRet;
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const ppu::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
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
    // The maximum vector size is 128 bits on ppu GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const ppu::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const ppu::TargetInfo &targetInfo, int computeCapability,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        computeCapability(computeCapability),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    unsigned vecOrig = vec;
    if (llMask) {
      LLVM_DEBUG(DBGS() << "vec = " << vec
                        << " mask_alignment = " << getMaskAlignment(mask));
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      LLVM_DEBUG(llvm::dbgs() << " vec = " << vec << '\n');
    }

    if (vec == 1 && numElems > 1) {
      int maskValue = !llMask ? -1 : getMaskAlignment(mask);
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " origin vec = " << vecOrig
                       << " numElems = " << numElems << " mask is " << maskValue
                       << "\n";
    }
    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    // Load redundantly in all dims except reg
    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    LDBG("LoadOp numElems = " << numElems << " vec = " << vec
                              << " valueElemNBits = " << valueElemNBits << " "
                              << op.getType());
    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (auto canonicalVecStart = getCanonicalIndex(vecStart, regMask);
          vecStart != canonicalVecStart) {
        // For redundant registers, refer back to the canonical load
        for (auto iVec = 0; iVec < vec; ++iVec) {
          loadedVals.push_back(loadedVals[canonicalVecStart + iVec]);
        }
        continue;
      }

      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      TIXBuilder tixBuilder;

      Value pred = mask ? maskElems[vecStart] : Value{};

      const std::string readConstraint =
          (width == 64) ? "l" : ((width == 32) ? "r" : "c");
      const std::string writeConstraint =
          (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

      // prepare asm operands
      auto *dstsOpr = tixBuilder.newListOperand();
      // If there is a `other` value, use it to init.
      bool init = other == nullptr;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        auto *opr = tixBuilder.newOperand(writeConstraint,
                                          init); // =r operations
        dstsOpr->listAppend(opr);
      }

      if (other) {
        for (size_t ii = 0; ii < nWords; ++ii) {
          // TIX doesn't support mov.u8, so we need to use mov.u16
          TIXInstr &mov =
              tixBuilder.create("ppu.mov")->o("u" + std::to_string(movWidth));

          size_t size = width / valueElemNBits;

          auto vecTy = LLVM::getVectorType(valueElemTy, size);
          Value v = b.undef(vecTy);
          for (size_t s = 0; s < size; ++s) {
            Value falseVal = otherElems[vecStart + ii * size + s];
            Value sVal = createIndexAttrConstant(
                rewriter, loc, typeConverter->getIndexType(), s);
            v = b.insert_element(vecTy, v, falseVal, sVal);
          }
          v = b.bitcast(v, IntegerType::get(getContext(), width));

          TIXInstr::Operand *opr{};

          if (otherIsSplatConstInt) {
            int64_t replicatedSplatVal = 0;
            for (size_t s = 0; s < movWidth; s += valueElemNBits) {
              replicatedSplatVal |= splatVal << s;
            }
            opr = tixBuilder.newConstantOperand(replicatedSplatVal);
          } else
            opr = tixBuilder.newOperand(v, readConstraint);

          mov(dstsOpr->listGet(ii), opr);
        }
      }

      auto *addrOpr =
          tixBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

      // Create L2 cache policy register if needed
      Value l2PolicyReg =
          createCachePolicy(op.getEvict(), rewriter, loc, computeCapability);

      bool isSharedPtr = isSharedMemoryPointer(ptrElems[vecStart]);

      // Define the instruction opcode
      auto &ld = tixBuilder.create("ppu.ld")
                     ->o("volatile", op.getIsVolatile())
                     .o("global", !isSharedPtr)
                     .o("shared", isSharedPtr)
                     .o("ca", !isSharedPtr &&
                              op.getCache() == triton::CacheModifier::CA)
                     .o("cg", !isSharedPtr &&
                              op.getCache() == triton::CacheModifier::CG)
                     .o("L1::evict_first",
                        !isSharedPtr &&
                            op.getEvict() == triton::EvictionPolicy::EVICT_FIRST)
                     .o("L1::evict_last",
                        !isSharedPtr &&
                            op.getEvict() == triton::EvictionPolicy::EVICT_LAST)
                     .o("L2::cache_hint", !isSharedPtr && l2PolicyReg != Value())
                     .v(nWords)
                     .b(width);

      TIXBuilder::Operand *evictOpr = nullptr;
      if (l2PolicyReg)
        evictOpr = tixBuilder.newOperand(l2PolicyReg, "l");

      if (!evictOpr)
        ld(dstsOpr, addrOpr).maybePredicate(pred, "b");
      else
        ld(dstsOpr, addrOpr, evictOpr).maybePredicate(pred, "b");

      // Create inline ASM signature
      SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
      Type retTy = retTys.size() > 1
                       ? LLVM::LLVMStructType::getLiteral(getContext(), retTys)
                       : retTys[0];

      Value ret = tixBuilder.launch(rewriter, loc, retTy);

      // Extract and store return values
      SmallVector<Value> rets;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr;
        if (isa<LLVM::LLVMStructType>(retTy)) {
          curr = b.extract_val(IntegerType::get(getContext(), width), ret, ii);
        } else {
          curr = ret;
        }
        curr = b.bitcast(
            curr, LLVM::getVectorType(valueElemTy, width / valueElemNBits));
        rets.push_back(curr);
      }
      int tmp = width / valueElemNBits;
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, typeConverter->getIndexType(), ii % tmp);
        Value loaded = b.extract_element(valueElemTy, rets[ii / tmp], vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }

  int computeCapability;
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const ppu::TargetInfo &targetInfo, int computeCapability,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        computeCapability(computeCapability),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    unsigned vecOrig = vec;
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    if (vec == 1 && elemsPerThread > 1) {
      int mask = !llMask ? -1 : getMaskAlignment(op.getMask());
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " origin vec = " << vecOrig
                       << " elemsPerThread = " << elemsPerThread << " mask is "
                       << mask << "\n";
    }

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    const int numVecs = elemsPerThread / vec;
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = b.undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = b.sext(i8_ty, elem);
          elem = b.bitcast(elem, valueElemTy);

          llWord = b.insert_element(wordTy, llWord, elem, b.i32_val(elemIdx));
        }
        llWord = b.bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      // Prepare the TIX inline asm.
      TIXBuilder tixBuilder;
      auto *asmArgList = tixBuilder.newListOperand(asmArgs);

      Value pred = threadPred;
      if (llMask) {
        auto mask = maskElems[vecStart];
        pred = maybeAnd(rewriter, loc, pred, mask);
      }

      auto *asmAddr =
          tixBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

      // Create L2 cache policy register if needed
      Value l2PolicyReg =
          createCachePolicy(op.getEvict(), rewriter, loc, computeCapability);

      bool isSharedStore = isSharedMemoryPointer(ptrElems[vecStart]);

      auto &tixStoreInstr =
          tixBuilder.create("ppu.st")
              ->o("global", !isSharedStore)
              .o("shared", isSharedStore)
              .o("wb", !isSharedStore &&
                       op.getCache() == triton::CacheModifier::WB)
              .o("cg", !isSharedStore &&
                       op.getCache() == triton::CacheModifier::CG)
              .o("cs", !isSharedStore &&
                       op.getCache() == triton::CacheModifier::CS)
              .o("wt", !isSharedStore &&
                       op.getCache() == triton::CacheModifier::WT)
              .o("L1::evict_first",
                 !isSharedStore &&
                     op.getEvict() == triton::EvictionPolicy::EVICT_FIRST)
              .o("L1::evict_last",
                 !isSharedStore &&
                     op.getEvict() == triton::EvictionPolicy::EVICT_LAST)
              .o("L2::cache_hint", !isSharedStore && l2PolicyReg != Value())
              .v(nWords)
              .b(width);

      TIXBuilder::Operand *evictOpr = nullptr;
      if (l2PolicyReg)
        evictOpr = tixBuilder.newOperand(l2PolicyReg, "l");

      if (!evictOpr)
        tixStoreInstr(asmAddr, asmArgList).maybePredicate(pred, "b");
      else
        tixStoreInstr(asmAddr, asmArgList, evictOpr).maybePredicate(pred, "b");

      auto asmReturnTy = void_ty(ctx);
      tixBuilder.launch(rewriter, loc, asmReturnTy);
    }
    rewriter.eraseOp(op);
    return success();
  }

  int computeCapability;
};

struct AsyncAIUCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  PPUAIUV1Conversion(triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type llvmElemTy =
        typeConverter->convertType(op.getResult().getType().getElementType());

    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    auto voidTy = void_ty(op->getContext());

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = ttg::lookupNumWarps(op);
    int warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpID = rewriter.create<ppugpu::WarpIdOp>(loc);
    warpID = LLVM::PPU::toUniformB32(loc, rewriter, warpID);

    int elementSizeInBytes =
        op.getResult().getType().getElementType().getIntOrFloatBitWidth() / 8;
    int totalNumElements = product(op.getResult().getType().getShape());
    int64_t size = totalNumElements * elementSizeInBytes;
    int rank = op.getCoord().size();

    /* AIU splite strategy
     */
    auto tensorType = op.getResult().getType();
    auto tileShape = tensorType.getShape();
    size_t tRank = tileShape.size();
    unsigned tileC = tileShape[tRank - 1];
    unsigned tileW = tileShape[tRank - 2];
    auto resSharedLayout =
        cast<ttg::PPUAIUSharedEncodingAttr>(tensorType.getEncoding());
    auto aiuLoad = resSharedLayout.getAIUStrategy();
    auto order = resSharedLayout.getOrder();

    unsigned cubeCElems = aiuLoad[0];
    unsigned cubeWElems = aiuLoad[1];
    unsigned cubeElems = cubeWElems * cubeCElems;
    unsigned warpCopyC = aiuLoad[2];
    unsigned warpCopyW = aiuLoad[3];
    unsigned warpW = numWarps / warpCopyC;
    Value warpIdxM = b.udiv(warpID, b.i32_val(warpCopyC));
    Value warpIdxN = b.urem(warpID, b.i32_val(warpCopyC));

    // warpN is eaqual with warpCopyC, warpW might be redundant
    bool isNeedPred = warpW > warpCopyW;
    Value pred = b.icmp_ult(warpIdxM, b.i32_val(warpCopyW));

    Value tensorShapeY = b.i32_val(1);
    Value tensorShapeX = op.getShape()[0];
    Value tensorShapeZ = op.getShape()[1];
    Value xCoord = op.getCoord()[0];
    Value zCoord = op.getCoord()[1];
    if (order[rank - 1] != 0) {
      tileC = tileShape[tRank - 2];
      tileW = tileShape[tRank - 1];
      tensorShapeX = op.getShape()[1];
      tensorShapeZ = op.getShape()[0];
      xCoord = op.getCoord()[1];
      zCoord = op.getCoord()[0];
    }

    Value yOffset = b.i32_val(0);
    Value cubeH = b.i32_val(1);
    Value cubeW = b.i32_val(cubeWElems);
    Value cubeZ = b.i32_val(cubeCElems);
    Value cubeN = b.i32_val(1);
    unsigned channelElemsPerCTA = cubeCElems * warpCopyC;
    unsigned CopyElemsPerCTA = cubeElems * warpCopyC * warpCopyW;

    unsigned numCopies = tileC / channelElemsPerCTA;
    //@$0
    std::string aiuInst =
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill." +
        std::to_string(rank) +
        "d.b16 [$0], [$1], {$2, $3, $4, $5, $6, $7}, {$8, $9, $10, $11};";
    if (isNeedPred) {
      aiuInst =
          "@$0 ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill." +
          std::to_string(rank) +
          "d.b16 [$1], [$2], {$3, $4, $5, $6, $7, $8}, {$9, $10, $11, $12};";
    }
    for (int copyIdx = 0; copyIdx < numCopies; copyIdx++) {
      Value xOffset = b.add(xCoord, b.mul(warpIdxM, b.i32_val(cubeWElems)));
      Value copyZOff = b.mul(b.i32_val(copyIdx), b.i32_val(channelElemsPerCTA));
      Value warpZOff = b.mul(warpIdxN, b.i32_val(cubeCElems));
      Value zOffset = b.add(zCoord, b.add(copyZOff, warpZOff));

      ::mlir::triton::ppu::TIXBuilder tixBuilderAIU;
      Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value copyIdxVal = b.add(warpID, b.i32_val(copyIdx));
      Value shMemOffsetCopies = b.i32_val(CopyElemsPerCTA * copyIdx);
      Value shMemOffsetWarps = b.mul(warpID, b.i32_val(cubeElems));
      Value shMemOffset = b.add(shMemOffsetCopies, shMemOffsetWarps);
      Value shMemPtr =
          b.gep(elemPtrTy, llvmElemTy, dstMemObj.getBase(), shMemOffset);
      SmallVector<TIXBuilder::Operand *> operands;
      if (isNeedPred) {
        operands.push_back(tixBuilderAIU.newOperand(pred, "b"));
      }
      operands.push_back(tixBuilderAIU.newOperand(shMemPtr, "r"));
      operands.push_back(tixBuilderAIU.newOperand(adaptor.getSrc(), "l"));

      operands.push_back(tixBuilderAIU.newOperand(tensorShapeY, "r"));
      operands.push_back(tixBuilderAIU.newOperand(tensorShapeX, "r"));
      operands.push_back(tixBuilderAIU.newOperand(tensorShapeZ, "r"));
      operands.push_back(tixBuilderAIU.newOperand(yOffset, "r"));
      operands.push_back(tixBuilderAIU.newOperand(xOffset, "r"));
      operands.push_back(tixBuilderAIU.newOperand(zOffset, "r"));
      operands.push_back(tixBuilderAIU.newOperand(cubeH, "r"));
      operands.push_back(tixBuilderAIU.newOperand(cubeW, "r"));
      operands.push_back(tixBuilderAIU.newOperand(cubeZ, "r"));
      operands.push_back(tixBuilderAIU.newOperand(cubeN, "r"));

      auto &aiu = *tixBuilderAIU.create<>(aiuInst);
      aiu(operands, /*onlyAttachMLIRArgs=*/true);
      tixBuilderAIU.launch(rewriter, loc, voidTy);
    }

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }

  LogicalResult
  PPUAIUV2Conversion(triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp op,
                     OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type llvmElemTy =
        typeConverter->convertType(op.getResult().getType().getElementType());

    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    auto voidTy = void_ty(op->getContext());

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = ttg::lookupNumWarps(op);
    int warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpID = rewriter.create<ppugpu::WarpIdOp>(loc);
    warpID = LLVM::PPU::toUniformB32(loc, rewriter, warpID);

    int elementSizeInBytes =
        op.getResult().getType().getElementType().getIntOrFloatBitWidth() / 8;
    int totalNumElements = product(op.getResult().getType().getShape());
    int64_t size = totalNumElements * elementSizeInBytes;
    int rank = op.getCoord().size();

    /* AIU splite strategy
     */
    auto tensorType = op.getResult().getType();
    auto resSharedLayout =
        cast<ttg::PPUAIUSharedEncodingAttr>(tensorType.getEncoding());
    auto aiuLoad = resSharedLayout.getAIUStrategy();
    unsigned cubeCElems = aiuLoad[0];
    unsigned cubeWElems = aiuLoad[1];
    unsigned swizzledBytes = aiuLoad[4];
    unsigned swizzledElems = swizzledBytes / elementSizeInBytes;
    unsigned aiuFactor = swizzledElems / cubeCElems;
    assert((cubeCElems == 32 / elementSizeInBytes && aiuFactor == 2) ||
             (cubeCElems != 32 / elementSizeInBytes && aiuFactor == 1));
    unsigned cubeElems = cubeWElems * cubeCElems * aiuFactor;

    unsigned warpCopyC = aiuLoad[2];
    unsigned warpCopyW = aiuLoad[3];
    unsigned copyWarps = warpCopyC * warpCopyW;
    Value warpIdxM = b.udiv(warpID, b.i32_val(warpCopyC));
    Value warpIdxN = b.urem(warpID, b.i32_val(warpCopyC));
    Value swizzledMode = b.i32_val(0);
    if (swizzledBytes == 64) {
      swizzledMode = b.i32_val(1);
    }

    auto order = resSharedLayout.getOrder();
    auto tileShape = tensorType.getShape();
    size_t tRank = tileShape.size();
    unsigned tileC = tileShape[tRank - 1];
    unsigned tileW = tileShape[tRank - 2];
    Value dimN = b.i32_val(1);
    Value dimW = op.getShape()[0];
    Value dimC = op.getShape()[1];
    Value coordW = op.getCoord()[0];
    Value coordC = op.getCoord()[1];
    Value startN = b.i32_val(0);
    if (order[rank - 1] != 0) {
      tileC = tileShape[tRank - 2];
      tileW = tileShape[tRank - 1];
      dimW = op.getShape()[1];
      dimC = op.getShape()[0];
      coordW = op.getCoord()[1];
      coordC = op.getCoord()[0];
    }

    Value cubeW = b.i32_val(cubeWElems);
    Value cubeC = b.i32_val(cubeCElems);
    Value cubeN = b.i32_val(1);
    auto CopyElemsPerCTA = cubeElems * copyWarps;

    auto swizzledElemsCTA = swizzledElems * warpCopyC * aiuFactor;
    unsigned copies = std::max<unsigned>(1, tileC / swizzledElemsCTA);

    // two kinds of situation needs pred:
    // 1. warp used in copy is less than numWarps
    // 2. warp used in copy C will copy more thand tileC
    bool isNeedPred = false;
    if (copyWarps < numWarps) {
      isNeedPred = true;
    }
    if (tileC > copies * swizzledElemsCTA) {
      isNeedPred = true;
    }
    //@$0
    std::string aiuInst;
    std::string dtype = (elementSizeInBytes == 2) ? ".b16" : ".b8";
    aiuInst = "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl" + dtype +
              "[$0], [$1], {$2, $3, $4}, {$5, $6, $7}, {$8, $9}, {$10, $11, $12}, $13;";
    if(isNeedPred) {
      aiuInst = "@$0 ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl" + dtype +
                "[$1], [$2], {$3, $4, $5}, {$6, $7, $8}, {$9, $10}, {$11, $12, $13}, $14;";
    }
    Value predW = b.icmp_ult(warpIdxM, b.i32_val(warpCopyW));
    auto i32Ty = IntegerType::get(op.getContext(), 32);
    for (unsigned copyIdx = 0; copyIdx * swizzledElemsCTA < tileC; copyIdx++) {
      Value copiesLoc = b.i32_val(copyIdx * swizzledElemsCTA);
      Value warpsLoc = b.mul(warpIdxN, b.i32_val(swizzledElems));
      Value predC = b.icmp_ult(b.add(copiesLoc, warpsLoc), b.i32_val(tileC));
      Value pred = b.and_(predC, predW);

      Value tensorStrideW = b.mul(b.trunc(i32Ty, dimC), b.i32_val(elementSizeInBytes));
      Value tensorStrideN = b.mul(b.trunc(i32Ty, dimW), tensorStrideW);
#if 1
      Value startW = b.add(coordW, b.mul(warpIdxM, b.i32_val(cubeWElems)));
      Value copyZOff = b.mul(b.i32_val(copyIdx), b.i32_val(swizzledElemsCTA));
      Value warpZOff = b.mul(warpIdxN, b.i32_val(cubeCElems));
      Value startC = b.add(coordC, b.add(copyZOff, warpZOff));
#endif

      ::mlir::triton::ppu::TIXBuilder tixBuilderAIU;
      SmallVector<TIXBuilder::Operand *> operands;

      Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value copyIdxVal = b.add(warpID, b.i32_val(copyIdx));
      Value shMemOffsetCopies = b.i32_val(CopyElemsPerCTA * copyIdx);
      Value shMemOffsetWarps = b.mul(warpID, b.i32_val(cubeElems));
      Value shMemOffset = b.add(shMemOffsetCopies, shMemOffsetWarps);
      Value shMemPtr =
          b.gep(elemPtrTy, llvmElemTy, dstMemObj.getBase(), shMemOffset);

      if (isNeedPred) {
        operands.push_back(tixBuilderAIU.newOperand(pred, "b"));
      }
      operands.push_back(tixBuilderAIU.newOperand(shMemPtr, "r"));
      operands.push_back(tixBuilderAIU.newOperand(adaptor.getSrc(), "l"));

      operands.push_back(tixBuilderAIU.newOperand(dimC, "r"));
      operands.push_back(tixBuilderAIU.newOperand(dimW, "r"));
      operands.push_back(tixBuilderAIU.newOperand(dimN, "r"));
      operands.push_back(tixBuilderAIU.newOperand(cubeC, "r"));
      operands.push_back(tixBuilderAIU.newOperand(cubeW, "r"));
      operands.push_back(tixBuilderAIU.newOperand(cubeN, "r"));
      operands.push_back(tixBuilderAIU.newOperand(tensorStrideW, "r"));
      operands.push_back(tixBuilderAIU.newOperand(tensorStrideN, "r"));
      operands.push_back(tixBuilderAIU.newOperand(startC, "r"));
      operands.push_back(tixBuilderAIU.newOperand(startW, "r"));
      operands.push_back(tixBuilderAIU.newOperand(startN, "r"));
      operands.push_back(tixBuilderAIU.newOperand(swizzledMode, "r"));

      auto &aiu = *tixBuilderAIU.create<>(aiuInst);
      aiu(operands, /*onlyAttachMLIRArgs=*/true);
      tixBuilderAIU.launch(rewriter, loc, voidTy);
    }

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }

  LogicalResult
  matchAndRewrite(triton::ppu_gpu::AsyncAIUCopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getCache() == triton::CacheModifier::NONE &&
           "cache modifiers not supported yet.");
    assert(op.getEvict() == triton::EvictionPolicy::NORMAL &&
           "eviction policy not supported yet.");
    auto tensorType = op.getResult().getType();
    if (auto resSharedLayout =
            dyn_cast<ttg::PPUAIUSharedEncodingAttr>(tensorType.getEncoding())) {
      if (resSharedLayout.isPPU0010()) {
        return PPUAIUV1Conversion(op, adaptor, rewriter);
      } else if (resSharedLayout.isPPU0015()) {
        return PPUAIUV2Conversion(op, adaptor, rewriter);
      } else {
        assert(false && "Unsuppored AIU Load");
      }
    } else {
      assert(false && "Unsupported AIU shared layout");
    }
    return failure();
  }
};

void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (numCTAs == 1) {
    b.barrier();
  } else {
    assert(false && "Barrier is currently unsuppored for numCTAs > 1");
  }
}

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const ppu::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; i += 1) {
      if (auto canonicalStart = getCanonicalIndex(i, regMask);
          canonicalStart != i) {
        // For redundant registers, refer back to the canonical result
        resultVals[i] = resultVals[canonicalStart];
        continue;
      }

      Value casVal = valElements[i];
      Value casCmp = cmpElements[i];
      Value casPtr = ptrElements[i];
      TIXBuilder tixBuilderAtomicCAS;
      std::string tyId =
          valueElemNBits == 64 ? "l" : (valueElemNBits == 32 ? "r" : "h");
      auto *dstOpr = tixBuilderAtomicCAS.newOperand("=" + tyId, /*init=*/true);
      auto *ptrOpr = tixBuilderAtomicCAS.newAddrOperand(casPtr, "l");
      auto *cmpOpr = tixBuilderAtomicCAS.newOperand(casCmp, tyId);
      auto *valOpr = tixBuilderAtomicCAS.newOperand(casVal, tyId);
      auto &atom = *tixBuilderAtomicCAS.create("ppu.atom");
      auto sTy = "b" + std::to_string(valueElemNBits);
      std::string semStr;
      llvm::raw_string_ostream os(semStr);
      os << op.getSem();
      auto scope = stringifyMemSyncScope(op.getScope()).str();
      bool isSharedCAS = isSharedMemoryPointer(casPtr);
      atom.o("global", !isSharedCAS).o("shared", isSharedCAS)
          .o(semStr).o(scope).o("cas").o(sTy);
      atom(dstOpr, ptrOpr, cmpOpr, valOpr).maybePredicate(threadPred);

      if (tensorTy) {
        auto retType = valueElemTy;
        auto ret = tixBuilderAtomicCAS.launch(rewriter, loc, retType);
        resultVals[i] = ret;
      } else {
        auto old = tixBuilderAtomicCAS.launch(rewriter, loc, valueElemTy);
        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with mask = True store the result
        TIXBuilder tixBuilderStore;
        auto *dstOprStore = tixBuilderStore.newAddrOperand(atomPtr, "r");
        auto *valOprStore = tixBuilderStore.newOperand(old, tyId);
        auto &st = *tixBuilderStore.create("ppu.st");
        st.shared().o(sTy);
        st(dstOprStore, valOprStore).maybePredicate(threadPred);
        auto ASMReturnTy = void_ty(ctx);
        tixBuilderStore.launch(rewriter, loc, ASMReturnTy);
        createBarrier(rewriter, loc, numCTAs);
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const ppu::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  bool supportsVectorized(RMWOp opType, Type elementType) const {
    // vectorized atomics are only supported on SM90+,
    // and only for specific atomic ops (add, min, max).
    // Note that "packed types" like f16x2 are supported sm60+.
    if (!targetInfo.supportVectorizedAtomics()) {
      return false;
    }

    return opType == RMWOp::FADD &&
           (elementType.isF16() || elementType.isBF16() || elementType.isF32());
  }

  bool isPromotableToTIXLD(triton::AtomicRMWOp op) const {
    if (disableLDAcquireLowering)
      return false;

    Type valueTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(op.getType()));

    if (!valueTy.isIntOrFloat())
      return false;
    if (op.getSem() != triton::MemSemantic::ACQUIRE &&
        op.getSem() != triton::MemSemantic::RELAXED)
      return false;
    if (op.getScope() != triton::MemSyncScope::CTA &&
        op.getScope() != triton::MemSyncScope::GPU &&
        op.getScope() != triton::MemSyncScope::SYSTEM)
      return false;

    if (op.getAtomicRmwOp() != RMWOp::ADD && op.getAtomicRmwOp() != RMWOp::FADD)
      return false;
    if (isa<RankedTensorType>(op.getType()))
      return false;
    if (!op.getVal().getDefiningOp())
      return false;
    if (!isa<arith::ConstantOp>(op.getVal().getDefiningOp()))
      return false;

    auto constOp = cast<arith::ConstantOp>(op.getVal().getDefiningOp());
    if (!isa<FloatAttr>(constOp.getValueAttr()) &&
        !isa<IntegerAttr>(constOp.getValueAttr()))
      return false;

    if (auto attr = dyn_cast_or_null<FloatAttr>(constOp.getValueAttr()))
      if (!attr.getValue().isZero())
        return false;

    if (auto attr = dyn_cast_or_null<IntegerAttr>(constOp.getValueAttr()))
      if (!attr.getValue().isZero())
        return false;

    return true;
  }

public:
  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicRMWOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    const size_t valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // packed: e.g. packed=2 for f16x2
    // vec: e.g. .v2, .v4, .v8 version of atom instruction.
    unsigned vec, vecOrig;
    int numElems, packed;
    if (tensorTy) {
      vec = getVectorSize(ptr);
      if (llMask) {
        vec = std::min<unsigned>(vec, getMaskAlignment(op.getMask()));
      }
      vecOrig = vec;
      packed = 1;
      auto valTy = cast<RankedTensorType>(val.getType());
      if (!supportsVectorized(atomicRmwAttr, valTy.getElementType())) {
        packed =
            std::min<unsigned>(vecOrig, valTy.getElementType().isF16() ? 2 : 1);
        vec = 1;
      }
      numElems = tensorTy.getNumElements();
    } else {
      // scalar
      vec = 1;
      vecOrig = 1;
      numElems = 1;
      packed = 1;
    }
    assert((packed == 1 || vec == 1) && "packed or vec must be 1");

    if (vec * packed == 1 && numElems > 1)
      op->emitRemark() << "Warning: vectorization fails vec = " << vec
                       << " packed = " << packed << " origin vec = " << vecOrig
                       << " numElems = " << numElems;

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    auto packedTy = vec_ty(valueElemTy, packed);
    SmallVector<Value> resultVals(elemsPerThread);

    // Lower AtomicRMWOp to a ld.acquire if possible
    std::unordered_map<triton::MemSyncScope, triton::ppugpu::MemSyncScope>
        ScopeMap = {
            {triton::MemSyncScope::CTA, triton::ppugpu::MemSyncScope::CTA},
            {triton::MemSyncScope::GPU, triton::ppugpu::MemSyncScope::GPU},
            {triton::MemSyncScope::SYSTEM,
             triton::ppugpu::MemSyncScope::SYSTEM}};
    const bool doTIXLDPromotion = isPromotableToTIXLD(op) && vec == 1 &&
                                  packed == 1 && ScopeMap.count(op.getScope());

    for (size_t i = 0; i < elemsPerThread; i += vec * packed) {
      if (auto canonicalStart = getCanonicalIndex(i, regMask);
          canonicalStart != i) {
        // For redundant registers, refer back to the canonical result
        for (auto iVecPack = 0; iVecPack < vec * packed; ++iVecPack) {
          resultVals[i + iVecPack] = resultVals[canonicalStart + iVecPack];
        }
        continue;
      }

      Value rmwPtr = ptrElements[i];
      Value pred = llMask ? maybeAnd(rewriter, loc, threadPred, maskElements[i])
                          : threadPred;

      if (doTIXLDPromotion) {
        Type convertedValueTy =
            getTypeConverter()->convertType(getElementTypeOrSelf(op.getType()));
        auto loadAcquireOp = triton::ppugpu::LoadAcquireOp::create(
            rewriter, op.getLoc(), convertedValueTy, rmwPtr, pred,
            op.getSem() == triton::MemSemantic::ACQUIRE
                ? triton::ppugpu::MemSemantic::ACQUIRE
                : triton::ppugpu::MemSemantic::RELAXED,
            ScopeMap[op.getScope()]);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with rmwMask = True store the result
        targetInfo.storeShared(rewriter, loc, atomPtr, loadAcquireOp, pred);
        createBarrier(rewriter, loc, numCTAs);
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }

      // Let LLVM handle compare+swap loop; branch-based pred should be fine
      if (valueElemTy.isBF16() && getPPUComputeCapability(moduleOp) < 90) {
        // Lower atomic bin-op and sem to LLVM
        auto llvmAtomicBinOp = matchAtomicOp(atomicRmwAttr);
        auto llvmAtomicMemOrdering = getMemoryOrdering(op.getSem());

        // Generate dominating undef
        Value undefVal = b.undef(valueElemTy);

        // Create basic block and branch to handle mask
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Setup the BlockArgument to return the result
        endBlock->addArgument({valueElemTy}, {loc});

        // Enter into predicate block
        rewriter.setInsertionPointToEnd(curBlock);
        bool doesAtomicNeedMEM = !op.getResult().use_empty();

        // Setup for SMEM Sync case
        Value atomPtr = tensorTy || !doesAtomicNeedMEM
                            ? nullptr
                            : LLVM::getSharedMemoryBase(
                                  loc, rewriter, targetInfo, op.getOperation());
        LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock,
                               undefVal);

        // Codegen the atomic-rmw instruction(s)
        rewriter.setInsertionPointToEnd(atomicBlock);
        Value atom =
            LLVM::AtomicRMWOp::create(rewriter, loc, *llvmAtomicBinOp, rmwPtr,
                                      valElements[i], *llvmAtomicMemOrdering,
                                      StringRef("device"))
                .getResult();
        // Handle the 2 bf16 case
        if (packed == 2 && valueElemNBits == 16) {
          Value atom2 = LLVM::AtomicRMWOp::create(
                            rewriter, loc, *llvmAtomicBinOp, ptrElements[i + 1],
                            valElements[i + 1], *llvmAtomicMemOrdering,
                            StringRef("device"))
                            .getResult();
          auto vecTy = vec_ty(valueElemTy, vec);
          auto tmp =
              b.insert_element(vecTy, b.undef(vecTy), atom, b.i32_val(0));
          atom = b.insert_element(vecTy, tmp, atom2, b.i32_val(1)).getResult();
        }

        if (tensorTy) {
          // Return from predicated block
          LLVM::BrOp::create(rewriter, loc, atom, endBlock);

          // Recover values from predicated block
          rewriter.setInsertionPointToStart(endBlock);
          Value ret = endBlock->getArgument(0);
          if (vec > 1) {
            for (unsigned ii = 0; ii < vec; ++ii) {
              resultVals[i + ii] = b.extract_val(valueElemTy, ret, ii);
            }
          } else if (packed > 1) {
            for (unsigned ii = 0; ii < packed; ++ii) {
              resultVals[i + ii] =
                  b.extract_element(valueElemTy, ret, b.i32_val(ii));
            }
          } else {
            resultVals[i] = ret;
          }
        } else {
          if (!doesAtomicNeedMEM) {
            LLVM::BrOp::create(rewriter, loc, atom, endBlock);
            rewriter.eraseOp(op);
            // if type isn't a tensor and there is no need to write to SMEM then
            // we are done here
            return success();
          }

          // Commit values from predicated block to SMEM and return from
          // predicate block
          // Note: there is no need to use the BlockArgument here because
          //       the value is recovered from SMEM in the !tensorTy case
          b.store(atom, atomPtr);
          LLVM::BrOp::create(rewriter, loc, atom, endBlock);

          // Recover values from predicated block (from SMEM)
          rewriter.setInsertionPointToStart(endBlock);
          b.barrier();
          Value ret = b.load(valueElemTy, atomPtr);
          rewriter.replaceOp(op, {ret});
          return success();
        }
        continue;
      }

      std::string sTy;
      TIXBuilder tixBuilderAtomicRMW;
      // 16-bit -> "h", 32-bit -> "r", 64-bit -> "l"
      std::string tyId =
          getRegisterSizeCode(valueElemNBits * packed, /*is_float=*/false);

      TIXBuilder::Operand *dstOpr;
      if (vec > 1) {
        dstOpr = tixBuilderAtomicRMW.newListOperand();
        for (unsigned ii = 0; ii < vec; ++ii) {
          dstOpr->listAppend(
              tixBuilderAtomicRMW.newOperand("=" + tyId, /*init=*/true));
        }
      } else {
        dstOpr = tixBuilderAtomicRMW.newOperand("=" + tyId, /*init=*/true);
      }

      auto *ptrOpr = tixBuilderAtomicRMW.newAddrOperand(rmwPtr, "l");

      TIXBuilder::Operand *valOpr;
      if (vec > 1) {
        valOpr = tixBuilderAtomicRMW.newListOperand();
        for (unsigned ii = 0; ii < vec; ++ii) {
          valOpr->listAppend(
              tixBuilderAtomicRMW.newOperand(valElements[i + ii], tyId));
        }
      } else if (packed > 1) {
        Value rmwVal = b.undef(packedTy);
        for (int ii = 0; ii < packed; ++ii) {
          rmwVal = b.insert_element(packedTy, rmwVal, valElements[i + ii],
                                    b.i32_val(ii));
        }
        valOpr = tixBuilderAtomicRMW.newOperand(rmwVal, tyId);
      } else {
        valOpr = tixBuilderAtomicRMW.newOperand(valElements[i], tyId);
      }

      auto scope = stringifyMemSyncScope(op.getScope()).str();
      bool isSharedRMW = isSharedMemoryPointer(rmwPtr);
      auto &atom = tixBuilderAtomicRMW.create("ppu.atom")
                       ->o("global", !isSharedRMW)
                       .o("shared", isSharedRMW)
                       .o(scope);
      auto rmwOp = stringifyRMWOp(atomicRmwAttr).str();
      auto sBits = std::to_string(valueElemNBits);
      switch (atomicRmwAttr) {
      case RMWOp::AND:
        sTy = "b" + sBits;
        break;
      case RMWOp::OR:
        sTy = "b" + sBits;
        break;
      case RMWOp::XOR:
        sTy = "b" + sBits;
        break;
      case RMWOp::ADD:
        sTy = "u" + sBits;
        break;
      case RMWOp::FADD:
        rmwOp = "add";
        rmwOp += (valueElemNBits == 16 ? ".noftz" : "");
        sTy = (valueElemTy.isBF16() ? "bf" : "f") + sBits;
        sTy += (packed == 2 && valueElemNBits == 16) ? "x2" : "";
        break;
      case RMWOp::MAX:
        sTy = "s" + sBits;
        break;
      case RMWOp::MIN:
        sTy = "s" + sBits;
        break;
      case RMWOp::UMAX:
        rmwOp = "max";
        sTy = "u" + sBits;
        break;
      case RMWOp::UMIN:
        rmwOp = "min";
        sTy = "u" + sBits;
        break;
      case RMWOp::XCHG:
        sTy = "b" + sBits;
        break;
      default:
        return failure();
      }
      std::string semStr;
      llvm::raw_string_ostream os(semStr);
      os << op.getSem();
      atom.o(semStr).o(rmwOp).v(vec).o(sTy);
      if (tensorTy) {
        atom(dstOpr, ptrOpr, valOpr).maybePredicate(pred);
        Type retType;
        if (vec > 1) {
          SmallVector<Type> retTys(vec, valueElemTy);
          retType = struct_ty(retTys);
        } else if (packed > 1) {
          retType = packedTy;
        } else {
          retType = valueElemTy;
        }

        Value ret = tixBuilderAtomicRMW.launch(rewriter, loc, retType);

        if (vec > 1) {
          for (unsigned ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] = b.extract_val(valueElemTy, ret, ii);
          }
        } else if (packed > 1) {
          for (unsigned ii = 0; ii < packed; ++ii) {
            resultVals[i + ii] =
                b.extract_element(valueElemTy, ret, b.i32_val(ii));
          }
        } else {
          resultVals[i] = ret;
        }
      } else {
        auto ASMReturnTy = void_ty(ctx);
        atom(dstOpr, ptrOpr, valOpr).maybePredicate(pred);
        auto old = tixBuilderAtomicRMW.launch(rewriter, loc, valueElemTy);
        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with rmwMask = True store the result
        targetInfo.storeShared(rewriter, loc, atomPtr, old, pred);
        createBarrier(rewriter, loc, numCTAs);
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const ppu::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // %src
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    // We assume other = 0, see XXX(Keren) below
    // %other
    // SmallVector<Value> otherElems;
    // if (llOther) {
    //   otherElems = unpackLLElements(loc, llOther, rewriter);
    //   assert(srcElems.size() == otherElems.size());
    // }

    // zip(src, mask)
    SmallVector<Value> vals;
    auto ptrTy = srcElems[0].getType();
    auto structTy =
        LLVM::LLVMStructType::getLiteral(ctx, ArrayRef<Type>{ptrTy, i1_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = LLVM::UndefOp::create(rewriter, loc, structTy);
      packedArr = b.insert_val(packedArr, srcElems[i], 0);
      auto maskElem = llMask ? maskElems[i] : b.false_val();
      packedArr = b.insert_val(packedArr, maskElem, 1);
      vals.push_back(packedArr);
    }

    // Remove broadcasted registers
    auto srcLayout = ttg::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    vals = removeBroadcastSrc.apply(vals);

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned maxVec = getContiguity(op.getSrc());
    if (mask) {
      maxVec = std::min(maxVec, getMaskAlignment(mask));
    }
    // If the op has a contiguity hint use it to increase the vector size.
    maxVec = std::max(maxVec, op.getContiguity());
    // The maximum vector size is 128 bits on ppu GPUs.
    maxVec = std::min(maxVec, 128 / resElemTy.getIntOrFloatBitWidth());

    int vecBytes = maxVec * resElemTy.getIntOrFloatBitWidth() / 8;
    if (vecBytes < 4) {
      return emitError(loc, "cp.async does not support transfers smaller than "
                            "4 bytes; calculated this as ")
             << vecBytes << " bytes";
    }
    assert(vecBytes == 16 || vecBytes == 8 || vecBytes == 4);

    auto freeVarMasks = getFreeVariableMasks(srcTy);
    // NOTE(@peterbell10): We load redundant data on different CTAs, so the data
    // is available in each CTAs respective shared memory. Otherwise, we would
    // need an additional broadcast step to copy the data between CTAs.
    freeVarMasks[str_attr("block")] = 0;
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    auto emitCpAsync = [&b, threadPred, ptrTy, hasMask = bool(llMask)](
                           RewriterBase &rewriter, Location loc,
                           ArrayRef<Value> vals, Value shmemAddr, int startIdx,
                           VectorType vecTy) -> SmallVector<Value> {
      assert(isa<VectorType>(vecTy));
      auto *ctx = rewriter.getContext();
      auto elemTy = vecTy.getElementType();
      auto nBytes = vecTy.getNumElements() * elemTy.getIntOrFloatBitWidth() / 8;
      assert(nBytes == 16 || nBytes == 8 || nBytes == 4);
      // Tune CG and CA.
      CacheModifier srcCacheModifier =
          nBytes == 16 ? CacheModifier::CG : CacheModifier::CA;

      auto structElem = vals[startIdx];
      auto srcElem = b.extract_val(ptrTy, structElem, 0);
      auto maskElem = b.extract_val(i1_ty, structElem, 1);

      TIXBuilder tixBuilder;
      auto &copyAsyncOp =
          *tixBuilder.create<TIXCpAsyncLoadInstr>(srcCacheModifier);
      auto *dstOperand = tixBuilder.newAddrOperand(shmemAddr, "r");
      auto *srcOperand = tixBuilder.newAddrOperand(srcElem, "l");
      auto *copySize = tixBuilder.newConstantOperand(nBytes);
      auto *srcSize = copySize;
      if (hasMask) {
        // We don't use predicate in this case, setting src-size to 0
        // if there's any mask. cp.async will automatically fill the
        // remaining slots with 0 if cp-size > src-size.
        // XXX(Keren): Always assume other = 0 for now.
        // When 'other != 0' is supported, we will need to fold the
        // op.getMask() and redundantDataMask() into the same predicate, the
        // way it is done for LoadOp.
        auto selectOp = b.select(maskElem, b.i32_val(nBytes), b.i32_val(0));
        srcSize = tixBuilder.newOperand(selectOp, "r");
      }
      copyAsyncOp(dstOperand, srcOperand, copySize, srcSize)
          .maybePredicate(threadPred);
      tixBuilder.launch(rewriter, loc, void_ty(ctx));
      return {};
    };

    // %dst
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto smemLayout = ttg::toLinearLayout(dstTy);
    auto cvt = srcLayout.invertAndCompose(smemLayout);
    if (!cvt.isTrivialOver({str_attr("block")})) {
      return emitError(loc,
                       "cp.async does not support non-trivial block dimension");
    }
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    auto maskSpanAffineOffset = SharedMemoryObject::getMaskSpanOffsets(dstTy);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    lowerLdSt(
        loc, ctx, cvt, vals, resElemTy, smemObj.getBase(),
        [](Value v) { return v; }, affineOffset, maskSpanAffineOffset, laneId,
        warpId, rewriter, targetInfo, maxVec, emitCpAsync);

    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          IntegerType::get(op.getContext(), 32),
                                          rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

static LinearLayout
getMsgToUnpackedOffsetLayout(const LinearLayout &packedLayout,
                             ttg::MemDescType ty) {
  auto isFp4Padded =
      cast<NVMMASharedEncodingAttr>(ty.getEncoding()).getFp4Padded();
  if (!isFp4Padded) {
    return packedLayout;
  }
  auto ctx = ty.getContext();
  auto rank = ty.getRank();
  auto kMsg = str_attr("msg");
  auto kLastDim = str_attr("dim" + Twine(rank - 1));
  // Multiply to offset by 2 in the last dimension
  auto unpackLayout = LinearLayout::zeros1D(1, kMsg, kLastDim, 2);
  return unpackLayout * packedLayout;
}

static LinearLayout getUnswizzledLayout(triton::gpu::MemDescType type) {
  auto mmaEncoding = dyn_cast<NVMMASharedEncodingAttr>(type.getEncoding());
  if (!mmaEncoding) {
    assert(isa<ttg::SwizzledSharedEncodingAttr>(type.getEncoding()));
    return ttg::toLinearLayout(type);
  }
  assert(type.getShape() == type.getAllocShape().take_back(type.getRank()));
  return ttg::nvmmaSharedToLinearLayout(
      type.getShape(), cast<NVMMASharedEncodingAttr>(type.getEncoding()),
      /*disableSwizzle=*/true);
}


struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto num = op->getAttrOfType<IntegerAttr>("num");
    NVVM::CpAsyncWaitGroupOp::create(rewriter, loc, num);

    // Drop the result token.
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    NVVM::CpAsyncCommitGroupOp::create(rewriter, loc);

    // Drop the result token.
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};
} // namespace

void mlir::triton::ppu::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    int computeCapability, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<AsyncCopyGlobalToLocalOpConversion, AtomicCASOpConversion,
               AtomicRMWOpConversion>(typeConverter, targetInfo,
                                      axisInfoAnalysis, benefit);
  patterns.add<LoadOpConversion, StoreOpConversion>(
      typeConverter, targetInfo, computeCapability, axisInfoAnalysis, benefit);
  patterns.add<AsyncAIUCopyGlobalToLocalOpConversion>(typeConverter, benefit);
  patterns.add<AsyncCommitGroupOpConversion, AsyncWaitOpConversion>(
      typeConverter, benefit);
}
