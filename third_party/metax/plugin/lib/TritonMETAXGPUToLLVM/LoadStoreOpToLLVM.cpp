#include "TargetInfo.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;

namespace {

Value maybeAnd(RewriterBase &rewriter, Location loc, Value a, Value b) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  if (a && b) {
    return tb.and_(a, b);
  }
  return a ? a : b;
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask. The predicate may be null to indicate no
// predication is required.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const METAX::TargetInfo &targetInfo) {
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

  Value pred = b.true_val();
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

void getCopyAsyncSwizzledGvmPtrs(RewriterBase &rewriter, Operation *op,
                                 Type srcElemTy, SmallVector<Value> &swiGvmPtrs,
                                 ArrayRef<Value> srcElems,
                                 ArrayRef<Value> colOffsets, unsigned vec) {
  auto loc = op->getLoc();
  TritonLLVMOpBuilder b(loc, rewriter);
  auto srcPtrTy = ptr_ty(rewriter.getContext(), 1);
  for (int i = 0; i < srcElems.size(); i++) {
    auto colOffset = colOffsets[i / vec];
    Value swiGvmPtr = b.gep(srcPtrTy, srcElemTy, srcElems[i], colOffset);
    swiGvmPtrs[i] = swiGvmPtr;
  }
  return;
}

void appendIntrinsicModifer(std::string &str, int vec, Type elemType) {
  str += ".";
  if (vec > 1) {
    str += "v";
    str += std::to_string(vec);
  }
  if (elemType.isF32()) {
    str += "f32";
  } else if (elemType.isF64()) {
    str += "f64";
  } else if (elemType.isF16()) {
    str += "f16";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 64) {
    str += "i64";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 32) {
    str += "i32";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 16) {
    str += "i16";
  } else if (isa<IntegerType>(elemType) &&
             elemType.getIntOrFloatBitWidth() == 8) {
    str += "i8";
  } else {
    assert(false && "Intrinsic Load unsupported data type");
  }
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const METAX::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getContiguity(ptr);
  }

  unsigned getThreadConstRepeatTimes(Value ptr) const {
    if (getenv("TRITON_DISABLE_CONSTANCY_LOAD_LAYOUT_OPT")) {
      return 1;
    }
    // constancy > 1 and sizePerThread > 1 and (sizePerThread % constancy == 0)
    // can return value > 1
    auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto layout = tensorTy.getEncoding();
    auto order = triton::gpu::getOrder(tensorTy);
    long int sizePerThread = 0;
    if (auto blockEncoding =
            dyn_cast<triton::gpu::BlockedEncodingAttr>(layout)) {
      sizePerThread = blockEncoding.getSizePerThread()[order[0]];
    } else {
      auto srcEncodingLinear = triton::gpu::toLinearEncoding(tensorTy);
      sizePerThread = srcEncodingLinear.getSizePerThread()[order[0]];
    }
    auto constancy = axisInfo->getConstancy(order[0]);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(tensorTy)[order[0]];

    if (sizePerThread == 1 && (constancy > shapePerCTATile) &&
        (constancy % shapePerCTATile == 0)) {
      auto constantCTANums = constancy / shapePerCTATile;
      return constantCTANums;
    }
    // constancy > sizePerThread, return sizePerThread
    if ((constancy > sizePerThread) && (constancy % sizePerThread == 0))
      return sizePerThread;
    if (sizePerThread % constancy)
      return 1;
    return constancy;
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // The maximum vector size is 128 bits on METAX GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned
  getVectorSize(Value ptr,
                llvm::ArrayRef<int64_t> contiguityInterConstGroup) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto constRepeatPerThread = getThreadConstRepeatTimes(ptr);

    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    if (contiguity == 1 && constRepeatPerThread > 1) {
      auto layout = tensorTy.getEncoding();
      auto order = triton::gpu::getOrder(tensorTy);
      long int sizePerThread = 0;
      if (auto blockEncoding =
              dyn_cast<triton::gpu::BlockedEncodingAttr>(layout)) {
        sizePerThread = blockEncoding.getSizePerThread()[order[0]];
      } else {
        auto srcEncodingLinear = triton::gpu::toLinearEncoding(tensorTy);
        sizePerThread = srcEncodingLinear.getSizePerThread()[order[0]];
      }
      auto loadPerThread =
          std::max<unsigned>(sizePerThread / constRepeatPerThread, 1);
      contiguity = std::min<unsigned>(contiguityInterConstGroup[order[0]],
                                      loadPerThread);
    }
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);

    // The maximum vector size is 128 bits on METAX GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const METAX::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const METAX::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    // original valuesgetVectorSize
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
    auto contiguityPerConstArray = op.getContiguityInterConstGroup();
    unsigned vec;
    if (contiguityPerConstArray.size() > 0) {
      assert(contiguityPerConstArray[0] > 0 &&
             "contiguityPerConstArray can't be zero,contiguityPerConstArray is "
             "either null or bigger than zero");
      vec = getVectorSize(ptr, contiguityPerConstArray);
    } else {
      vec = getVectorSize(ptr);
    }
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    unsigned constRepeatPerThread = getThreadConstRepeatTimes(ptr);
    if (llMask) {
      LLVM_DEBUG(DBGS() << "vec = " << vec
                        << " mask_alignment = " << getMaskAlignment(mask));
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      constRepeatPerThread =
          std::min<unsigned>(constRepeatPerThread, getMaskAlignment(mask));
      LLVM_DEBUG(llvm::dbgs() << " vec = " << vec << '\n');
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
    bool otherIsSplatConst = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConst = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    float splatValfp = 0;
    if (other &&
        (isa<FloatType>(valueElemTy) || isa<IntegerType>(valueElemTy)) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<FloatType>(constAttr.getElementType())) {
      otherIsSplatConst = true;
      if (isa<Float64Type>(valueElemTy)) {
        splatValfp = static_cast<float>(
            constAttr.getSplatValue<APFloat>().convertToDouble());
      } else {
        splatValfp = constAttr.getSplatValue<APFloat>().convertToFloat();
      }
    }

    bool isOtherValid = other ? false : true;
    if (otherIsSplatConst && (splatVal == 0) && (splatValfp == 0))
      isOtherValid = true;

    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    const char *disableOptFlag = getenv("TRITON_DISABLE_LOAD_STORE_OPT");
    const char *disableLdgPred = getenv("TRITON_DISABLE_LDG_PREDICATOR");

    int stride;
    if (constRepeatPerThread > 1) {
      stride = vec * constRepeatPerThread;
    } else {
      stride = vec;
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    LDBG("LoadOp numElems = " << numElems << " vec = " << vec
                              << " valueElemNBits = " << valueElemNBits << " "
                              << op.getType());
    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += stride) {
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
      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value zeroVal = b.bitcast(b.int_val(valueElemNBits, 0), valueElemTy);
      if (!disableOptFlag && !op.getIsVolatile()) {
        if (!disableLdgPred && isOtherValid &&
            (totalWidth == 128 || totalWidth == 64 || totalWidth == 32 ||
             totalWidth == 16 || totalWidth == 8)) {
          // llvm.mxc.ldg.predicator
          Type retTy =
              vec > 1 ? vec_ty(valueElemTy, vec)
                      : valueElemTy; // RetType of llvm.mxc.ldg.predicator.f32
                                     // should be float
          Type retPtrTy = ptr_ty(rewriter.getContext(), 1);
          Value maskElem;
          maskElem = mask ? b.zext(IntegerType::get(rewriter.getContext(), 32),
                                   maskElems[vecStart])
                          : b.int_val(32, 1);
          // }
          std::string icmp = "llvm.mxc.icmp.i64.i32";
          StringRef icmpName(icmp);
          Value cmpModel = b.i32_val(34);
          SmallVector<Value> rangeValueCmp(3);
          rangeValueCmp[0] = maskElem;         // mask to be compared
          rangeValueCmp[1] = b.int_val(32, 0); // compare to 0
          rangeValueCmp[2] = cmpModel;         // compare model
          ValueRange icomValue(rangeValueCmp);

          Value xmask = mlir::LLVM::createBuiltinFunc<triton::LoadOp>(
              rewriter, loc, op, icmpName, i64_ty, icomValue);

          std::string ldgPredicator = "llvm.mxc.ldg.predicator";
          appendIntrinsicModifer(ldgPredicator, vec, valueElemTy);
          StringRef predictName(ldgPredicator);
          SmallVector<Value> rangeValueLdg(7);
          Value ldgPtr = b.bitcast(ptrElems[vecStart], retPtrTy);
          rangeValueLdg[0] = ldgPtr;       // global addr, default mtreg
          rangeValueLdg[1] = b.i32_val(0); // addr offset, immediate number

          rangeValueLdg[2] = xmask;           // mask
          rangeValueLdg[3] = b.int_val(1, 1); // enable return0
          if (std::getenv("TRITON_DISABLE_LDG_SADDR") != nullptr) {
            rangeValueLdg[4] = b.int_val(1, 0); // use saddr flag
          } else {
            rangeValueLdg[4] = b.int_val(1, 1); // use saddr flag
          }
          rangeValueLdg[5] = b.int_val(1, 0); // pred_neg flag
          rangeValueLdg[6] = b.int_val(1, 0); // disable async

          ValueRange ldgValue(rangeValueLdg);

          Value ldg_values = mlir::LLVM::createBuiltinFunc<triton::LoadOp>(
              rewriter, loc, op, predictName, retTy, ldgValue);
          if (vec == 1) {
            // loadedVals.push_back(ldg_values);
            for (int i = 0; i < constRepeatPerThread; i++) {
              loadedVals.push_back(ldg_values);
            }
          } else {
            for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
              Value curr = b.extract_element(valueElemTy, ldg_values,
                                             b.i32_val(elemIndex));
              for (int i = 0; i < constRepeatPerThread; i++) {
                loadedVals.push_back(curr);
              }
            }
          }
        } else { // llvm.mxc.load.global.async
          Type retTy = vec > 1 ? vec_ty(valueElemTy, vec) : valueElemTy;
          Type retPtrTy = ptr_ty(rewriter.getContext(), 1);
          auto loaded = rewriter.create<scf::IfOp>(
              loc, pred,
              [&](OpBuilder &builder, Location loc) {
                Value vec_values;
                vec_values =
                    b.load(retTy, b.bitcast(ptrElems[vecStart], retPtrTy));
                builder.create<mlir::scf::YieldOp>(loc,
                                                   ValueRange({vec_values}));
              },
              [&](OpBuilder &builder, Location loc) {
                Value vec_values = b.undef(retTy);
                if (vec == 1) {
                  Value otherVal = other ? otherElems[vecStart] : zeroVal;
                  vec_values = otherVal;
                } else {
                  for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
                    size_t elemOffset = vecStart + elemIndex;
                    Value otherVal = other ? otherElems[elemOffset] : zeroVal;
                    vec_values = b.insert_element(retTy, vec_values, otherVal,
                                                  b.i32_val(elemIndex));
                  }
                }
                builder.create<mlir::scf::YieldOp>(loc,
                                                   ValueRange({vec_values}));
              });
          Value ret = loaded->getResult(0);
          if (vec == 1) {
            // loadedVals.push_back(ret);
            for (int i = 0; i < constRepeatPerThread; i++) {
              loadedVals.push_back(ret);
            }
          } else {
            for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
              Value curr =
                  b.extract_element(valueElemTy, ret, b.i32_val(elemIndex));
              for (int i = 0; i < constRepeatPerThread; i++) {
                loadedVals.push_back(curr);
              }
            }
          }
        }
      } else {
        for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
          for (size_t wordElem = 0; wordElem < wordNElems; ++wordElem) {
            size_t elemOffset = vecStart + wordIdx * wordNElems + wordElem;
            if (mask) {
              auto loaded = rewriter.create<scf::IfOp>(
                  loc, pred,
                  [&](OpBuilder &builder, Location loc) {
                    auto loadVal = builder.create<LLVM::LoadOp>(
                        loc, valueElemTy, ptrElems[elemOffset], 0,
                        op.getIsVolatile());
                    builder.create<scf::YieldOp>(loc, ValueRange({loadVal}));
                  },
                  [&](OpBuilder &builder, Location loc) {
                    Value otherVal = other ? otherElems[elemOffset] : zeroVal;
                    builder.create<scf::YieldOp>(loc, ValueRange({otherVal}));
                  });
              loadedVals.push_back(loaded->getResult(0));
            } else {
              auto loadVal = rewriter.create<LLVM::LoadOp>(
                  loc, valueElemTy, ptrElems[elemOffset], 0,
                  op.getIsVolatile());
              loadedVals.push_back(loadVal);
            }
          }
        }
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const METAX::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
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
    MLIRContext *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    if (valueElemTy.isFloat(8)) {
      assert(false && "storeOp unsupported data type");
    }

    if (valueElemTy.isInteger(16)) {
      for (size_t elemIndex = 0; elemIndex < elemsPerThread; elemIndex++) {
        Value elem = valueElems[elemIndex];
        // check if value is constant and bf16
        if (auto constantOp =
                dyn_cast_or_null<LLVM::ConstantOp>(elem.getDefiningOp())) {
          auto valueAttr = constantOp.getValue();
          auto floatAttr = dyn_cast_or_null<mlir::FloatAttr>(valueAttr);
          if (floatAttr && floatAttr.getType().isBF16()) {
            // create new int16 constant value and replace the origin bf16 value
            llvm::APFloat bf16Value = floatAttr.getValue();
            llvm::APInt apInt = bf16Value.bitcastToAPInt();
            int16_t int16Value = static_cast<int16_t>(apInt.getZExtValue());
            mlir::Type int16Type = rewriter.getIntegerType(16);
            mlir::Attribute newValueAttr =
                rewriter.getI16IntegerAttr(int16Value);
            auto newConstantOp = rewriter.create<LLVM::ConstantOp>(
                constantOp.getLoc(), int16Type, newValueAttr);
            rewriter.replaceOp(constantOp, newConstantOp.getResult());
          }
        }
      }
    }

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    // guess this is mask for store broadcast dim
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
      if (totalWidth == 128 || totalWidth == 64 || totalWidth == 32) {
        Type retTy = vec > 1 ? vec_ty(valueElemTy, vec) : valueElemTy;
        Type retPtrTy = ptr_ty(rewriter.getContext(), 1);

        // global addr
        Value stgPtr = b.bitcast(ptrElems[vecStart], retPtrTy);
        // data to be stored
        Value vec_values = b.undef(retTy);
        if (vec > 1) {
          for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
            Value elem = valueElems[vecStart + elemIndex];
            vec_values =
                b.insert_element(retTy, vec_values, elem, b.i32_val(elemIndex));
          }
        } else {
          vec_values = valueElems[vecStart];
        }
        // mask
        Value maskVal =
            llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;
        // Value maskVal = llMask ? b.and_(mask, maskElems[vecStart]) : mask;
        Value maskElem =
            b.zext(IntegerType::get(rewriter.getContext(), 32), maskVal);
        std::string icmp = "llvm.mxc.icmp.i64.i32";
        StringRef icmpName(icmp);
        Value cmpModel = b.i32_val(34);
        SmallVector<Value> rangeValueCmp(3);
        rangeValueCmp[0] = maskElem;         // mask to be compared
        rangeValueCmp[1] = b.int_val(32, 0); // compare to 0
        rangeValueCmp[2] = cmpModel;         // compare model
        ValueRange icomValue(rangeValueCmp);

        Value xmask = mlir::LLVM::createBuiltinFunc<triton::StoreOp>(
            rewriter, loc, op, icmpName, i64_ty, icomValue);

        std::string stgPredicator = "llvm.mxc.stg.predicator";
        appendIntrinsicModifer(stgPredicator, vec, valueElemTy);
        StringRef stgPredictName(stgPredicator);

        SmallVector<Value> rangeValueStg(7);
        rangeValueStg[0] = stgPtr;       // global addr
        rangeValueStg[1] = b.i32_val(0); // global addr offset, immediate number
        rangeValueStg[2] = vec_values;   // data
        rangeValueStg[3] = xmask;        // mask
        if (std::getenv("TRITON_DISABLE_STG_SADDR") != nullptr) {
          rangeValueStg[4] = b.int_val(1, 0); // use saddr flag
        } else {
          rangeValueStg[4] = b.int_val(1, 1); // use saddr flag
        }
        rangeValueStg[5] = b.int_val(1, 0); // pred_neg flag
        rangeValueStg[6] = b.int_val(1, 0); // enable async
        ValueRange sdgValue(rangeValueStg);

        mlir::LLVM::createBuiltinFunc<triton::StoreOp>(
            rewriter, loc, op, stgPredictName, getVoidType(), sdgValue);
      } else {
        Value pred = threadPred;
        if (llMask) {
          auto mask = maskElems[vecStart];
          pred = maybeAnd(rewriter, loc, pred, mask);
        }
        Type retTy = vec_ty(valueElemTy, vec);
        Type retPtrTy = ptr_ty(ctx, 1);
        // when threadPred is null, set pred to default 1
        pred = pred ? pred : b.int_val(1, 1);
        rewriter.create<scf::IfOp>(
            loc, pred,
            [&](OpBuilder &builder, Location loc) {
              Value vec_values = b.undef(retTy);
              for (size_t elemIndex = 0; elemIndex < vec; elemIndex++) {
                Value elem = valueElems[vecStart + elemIndex];
                vec_values = b.insert_element(retTy, vec_values, elem,
                                              b.i32_val(elemIndex));
              }
              b.store(vec_values, b.bitcast(ptrElems[vecStart], retPtrTy));
              builder.create<mlir::scf::YieldOp>(loc);
            },
            nullptr);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

static LLVM::AtomicOrdering getMemoryOrdering(MemSemantic memOrdering) {
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return LLVM::AtomicOrdering::acq_rel;
  }
}

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::AtomicCASOp>::ConvertOpToLLVMPattern;

  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const METAX::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // extract relevant info from Module
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto TensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        TensorTy ? getTypeConverter()->convertType(TensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
    // tensor
    if (TensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = b.undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = b.insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];

      // as cmpx only support int type, here convert to int if it is not
      // fix "'llvm.cmpxchg' op operand #1 must be int" error
      Type intType;
      if (triton::type::isFloat(valueElemTy)) {
        if (valueElemTy.isF64()) {
          intType = rewriter.getI64Type(); // f64 → i64
        } else if (valueElemTy.isF32()) {
          intType = rewriter.getI32Type(); // f32 → i32
        } else if (valueElemTy.isF16() || valueElemTy.isBF16()) {
          intType = rewriter.getI16Type(); // f16/bf16 → i16
        } else {
          return failure();
        }

        casCmp = rewriter.create<LLVM::BitcastOp>(op.getLoc(), intType, casCmp);
        casVal = rewriter.create<LLVM::BitcastOp>(op.getLoc(), intType, casVal);
        Type intPtrType = LLVM::LLVMPointerType::get(
            rewriter.getContext(),
            cast<LLVM::LLVMPointerType>(casPtr.getType()).getAddressSpace());
        casPtr =
            rewriter.create<LLVM::BitcastOp>(op.getLoc(), intPtrType, casPtr);
      }

      // use op
      if (TensorTy) { // for tensor
        auto retType = vec == 1 ? valueElemTy : vecTy;
        // TODO: USE ATOMIC CAS OP on Tensor
        auto successOrdering = atomicMemOrdering;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef("device"));

        // Extract the new_loaded value from the pair.
        Value ret = nullptr;
        if (triton::type::isFloat(valueElemTy)) {
          ret = b.extract_val(intType, cmpxchg, i);
          ret = rewriter.create<LLVM::BitcastOp>(op.getLoc(), valueElemTy, ret);
        } else {
          ret = b.extract_val(valueElemTy, cmpxchg, i);
        }

        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret
                       : b.extract_element(valueElemTy, ret, b.i32_val(ii));
        }
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);

        auto tid = getThreadId(rewriter, loc);
        Value pred =
            maybeAnd(rewriter, loc, b.icmp_eq(tid, b.i32_val(i)), threadPred);
        rewriter.create<LLVM::CondBrOp>(loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef("device"));

        // Extract the new_loaded value from the pair.
        if (!op.getResult().use_empty()) {
          Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                    op.getOperation());
          Value newLoaded = nullptr;
          if (triton::type::isFloat(valueElemTy)) {
            newLoaded = b.extract_val(intType, cmpxchg, 0);
            newLoaded = rewriter.create<LLVM::BitcastOp>(
                op.getLoc(), valueElemTy, newLoaded);
          } else {
            newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          }
          b.store(newLoaded, atomPtr);
        }

        rewriter.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        // add arrive(0);
        StringRef funcName("llvm.mxc.arrive");
        auto loc = op.getLoc();
        Value num_value = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, 0);
        mlir::LLVM::createBuiltinFunc<triton::AtomicCASOp>(
            rewriter, loc, op, funcName, getVoidType(), {num_value});

        b.barrier();
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        b.barrier();
        rewriter.replaceOp(op, {ret});
      }
    }

    // replace op
    if (TensorTy) {
      Type structTy = getTypeConverter()->convertType(TensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::AtomicRMWOp>::ConvertOpToLLVMPattern;

  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const METAX::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  /// Try to match the mlir::triton::RMWOp to LLVM::AtomicBinOp.
  static std::optional<LLVM::AtomicBinOp> matchAtomicOp(RMWOp atomicOp) {
    switch (atomicOp) {
    case RMWOp::AND:
      return LLVM::AtomicBinOp::_and;
    case RMWOp::OR:
      return LLVM::AtomicBinOp::_or;
    case RMWOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case RMWOp::ADD:
      return LLVM::AtomicBinOp::add;
    case RMWOp::FADD:
      return LLVM::AtomicBinOp::fadd;
    case RMWOp::MAX:
      return LLVM::AtomicBinOp::max;
    case RMWOp::MIN:
      return LLVM::AtomicBinOp::min;
    case RMWOp::UMAX:
      return LLVM::AtomicBinOp::umax;
    case RMWOp::UMIN:
      return LLVM::AtomicBinOp::umin;
    case RMWOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    default:
      return std::nullopt;
    }
    llvm_unreachable("Invalid RMWOp");
  }

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto atomicRmwAttr = op.getAtomicRmwOp();
    Value ptr = op.getPtr();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    Value opResult = op.getResult();
    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();
    const size_t valueElemNbits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = b.int_val(1, 1);
    auto tid = getThreadId(rewriter, loc);
    bool needLdsStaging = !tensorTy && !opResult.use_empty();
    std::optional<Value> atomicSharedMemBase =
        op->hasAttr("allocation.offset") && needLdsStaging
            ? std::optional<Value>(getSharedMemoryBase(
                  loc, rewriter, targetInfo, op.getOperation()))
            : std::nullopt;

    mask = b.and_(mask, b.icmp_slt(b.mul(tid, b.i32_val(elemsPerThread)),
                                   b.i32_val(numElems)));

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);

    auto vecTy = vec_ty(valueElemTy, vec);
    auto retType = vec == 1 ? valueElemTy : vecTy;
    SmallVector<Value> resultVals(elemsPerThread);
    const bool f16v2 = vec == 2 && valueElemTy.isF16();
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(mask, maskElements[i]) : mask;

      Value undefVal = b.undef(retType);
      // Build blocks to bypass the atomic instruction for ~rmwMask.
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument({retType}, {loc});

      rewriter.setInsertionPointToEnd(curBlock);
      rewriter.create<LLVM::CondBrOp>(loc, rmwMask, atomicBlock, endBlock,
                                      undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      auto maybeKind = matchAtomicOp(atomicRmwAttr);
      Value atom;
      if (atomicRmwAttr == RMWOp::FADD && tensorTy &&
          memOrdering == MemSemantic::ACQUIRE_RELEASE &&
          tensorTy.getElementType().isBF16()) {
        std::string intrinsicName = "llvm.mxc.gvm.atomic.add.bf16.i16";

        Type retTy = valueElemTy;
        for (size_t j = 0; j < vec; ++j) {
          SmallVector<Value> rangeValue(2);
          rangeValue[0] = rmwPtr;
          rangeValue[1] = valElements[i + j];
          ValueRange atomicValue(rangeValue);
          Value atomic_val = mlir::LLVM::createBuiltinFunc<triton::AtomicRMWOp>(
              rewriter, loc, op, intrinsicName, i16_ty, atomicValue);
          atom = vec == 1 ? atomic_val
                          : b.insert_element(vecTy, b.undef(vecTy), atomic_val,
                                             b.i32_val(j));
        }
      } else {
        if (f16v2) {
          Value atom2 =
              rewriter
                  .create<LLVM::AtomicRMWOp>(
                      loc, *maybeKind, ptrElements[i + 1], valElements[i + 1],
                      atomicMemOrdering, StringRef("device"))
                  .getResult();
          auto tmp =
              b.insert_element(vecTy, b.undef(vecTy), atom, b.i32_val(0));
          atom = b.insert_element(vecTy, tmp, atom2, b.i32_val(1)).getResult();
        } else if (atomicRmwAttr == RMWOp::FADD && valueElemTy.isBF16()) {
          Type retTy = valueElemTy;
          std::string intrinsicName = "llvm.mxc.gvm.atomic.add.bf16.i16";
          for (size_t j = 0; j < vec; ++j) {
            SmallVector<Value> rangeValue(2);
            rangeValue[0] = rmwPtr;
            rangeValue[1] = b.bitcast(valElements[i + j], i16_ty);
            ValueRange atomicValue(rangeValue);
            Value atomic_val =
                mlir::LLVM::createBuiltinFunc<triton::AtomicRMWOp>(
                    rewriter, loc, op, intrinsicName, i16_ty, atomicValue);
            atomic_val = b.bitcast(atomic_val, valueElemTy);
            atom = vec == 1 ? atomic_val
                            : b.insert_element(vecTy, b.undef(vecTy),
                                               atomic_val, b.i32_val(j));
          }
        } else {
          atom = rewriter
                     .create<LLVM::AtomicRMWOp>(
                         loc, *maybeKind, rmwPtr, valElements[i],
                         atomicMemOrdering, StringRef("device"))
                     .getResult();
        }
      }
      if (!tensorTy && atomicSharedMemBase.has_value()) {
        Value atomPtr = *atomicSharedMemBase;
        Type atomPtrTy = ptr_ty(rewriter.getContext(), 3);
        b.store(atom, b.bitcast(atomPtr, atomPtrTy));
      }
      rewriter.create<LLVM::BrOp>(loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      Value retVal = endBlock->getArgument(0);
      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? retVal
                       : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
        }
      } else {
        if (!atomicSharedMemBase.has_value()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = *atomicSharedMemBase;
        b.barrier();
        Type atomPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value ret = b.load(valueElemTy, b.bitcast(atomPtr, atomPtrTy));
        b.barrier();
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const METAX::TargetInfo &targetInfo,
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
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = op.getSrc().getType();
    if (!isa<BlockedEncodingAttr>(srcTy.getEncoding()))
      return rewriter.notifyMatchFailure(op,
                                         "requires Blocked encoding for src");
    auto srcBlockedLayout = cast<BlockedEncodingAttr>(srcTy.getEncoding());

    auto dstTy = op.getResult().getType();
    auto resSharedLayout =
        dyn_cast<SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    if (!resSharedLayout)
      return rewriter.notifyMatchFailure(
          op, "requires swizzled shared encoding for dst");
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // %src
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);
    Type ptrTy = srcElems[0].getType();

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }
    // TODO: get mask alignment and update vec ?

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      // always assume other is 0 for now
      otherElems = unpackLLElements(loc, llOther, rewriter);
      assert(srcElems.size() == otherElems.size());
    }

    // %dst
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);

    // calculate load vector size
    // We don't use getVec() here because we are copying from memory to memory.
    // If contiguity > vector size, we can have one pointer maintaining the
    // start of the vector and the other pointer moving to the next vector.
    unsigned inVec = getContiguity(op.getSrc());
    // TODO: check MaskAlignment? swizzleTensorOp and ldg.bsm must have same
    // minVec value to ensure correctness if (mask) {
    //   inVec = std::min(inVec, getMaskAlignment(mask));
    // }
    inVec = std::min(inVec, 128 / resElemTy.getIntOrFloatBitWidth());

    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    if (maxPhase == 1) {
      minVec = inVec;
    }

    bool hasSwizzling = maxPhase > 1;
    // If dst shared layout has swizzle, move swizzle pattern to src pointers.
    // Because we only apply swizzling on contiguous dimension, and the
    // colOffset on contiguous dimension satisfy condition below:
    //    sharedSwizzledOffset = sharedNoSwizzledOffset + colOffset
    //    blockedSwizzledIndex = blockedNoSwizzledIndex + colOffset
    // So we can move swizzle pattern to src pointers by simply adding colOffset
    //
    // the src and dst layout of AsyncCopyGlobalToLocalOp
    //             blockedNoSwizzledIndex
    // |---tid0---|---tid1---|---tid2---|---tid3---|
    //             sharedSwizzledOffset
    // |---tid1---|---tid0---|---tid3---|---tid2---|
    //
    // after moving swizzle pattern to src pointers:
    //             blockedSwizzledIndex
    // |---tid1---|---tid0---|---tid3---|---tid2---|
    //             sharedNoSwizzledOffset
    // |---tid0---|---tid1---|---tid2---|---tid3---|

    auto noSwizzleDstTy = dstTy;
    SmallVector<Value> srcSwiAddrs = srcElems;

    if (hasSwizzling) {
      // 1. Get noSwizzleSharedLayout
      auto noSwizzleSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), outVec, 1, 1, resSharedLayout.getOrder(),
          resSharedLayout.getCTALayout());
      noSwizzleDstTy =
          MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                           noSwizzleSharedEnc, dstTy.getMemorySpace());
      // 2. Get colOffset (size == srcElems.size() / minVec)
      SmallVector<Value> colOffsets = triton::gpu::emitSwizzleColOffsets(
          rewriter, op, srcTy, dstTy.getShape(), resSharedLayout,
          noSwizzleSharedEnc, minVec);
      // 3. Get blockedSwizzledIndex
      getCopyAsyncSwizzledGvmPtrs(rewriter, op, resElemTy, srcSwiAddrs,
                                  srcElems, colOffsets, minVec);
    }

    // zip(srcSwiAddr, mask)
    SmallVector<Value> vals;
    auto structTy =
        LLVM::LLVMStructType::getLiteral(ctx, ArrayRef<Type>{ptrTy, i1_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = rewriter.create<LLVM::UndefOp>(loc, structTy);
      packedArr = b.insert_val(packedArr, srcSwiAddrs[i], 0);
      auto maskElem = llMask ? maskElems[i] : b.false_val();
      packedArr = b.insert_val(packedArr, maskElem, 1);
      vals.push_back(packedArr);
    }

    // 4. calculate cvt from swizzledBlockedLayout to noSwizzledSharedLayout
    // Build src to shared layout and remove broadcasted registers
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    srcSwiAddrs = removeBroadcastSrc.apply(srcSwiAddrs);

    LinearLayout noSwiSmemLayout = triton::gpu::toLinearLayout(noSwizzleDstTy);
    auto cvt = srcLayout.invertAndCompose(noSwiSmemLayout);
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    // 5. emit ldg.bsm using lowerLdSt
    // TODO: laneID???
    Value threadPred = emitRedundantThreadPredicate(getFreeVariableMasks(srcTy),
                                                    rewriter, loc, targetInfo);

    auto emitLdgBsm = [&b, op, threadPred, ptrTy, minVec](
                          RewriterBase &rewriter, Location loc,
                          ArrayRef<Value> vals, Value shmemAddr, int startIdx,
                          VectorType vecTy) -> SmallVector<Value> {
      assert(isa<VectorType>(vecTy));
      auto *ctx = rewriter.getContext();
      auto valueElemTy = vecTy.getElementType();
      auto structElem = vals[startIdx];
      auto gvmAddr = b.extract_val(ptrTy, structElem, 0);
      auto maskElem = b.extract_val(i1_ty, structElem, 1);

      Type gvmi8Ptr = ptr_ty(ctx, 1);
      Value gvmi8Addr = b.bitcast(gvmAddr, gvmi8Ptr);

      auto cond = maybeAnd(rewriter, loc, threadPred, maskElem);
      Value maskVal = b.zext(IntegerType::get(rewriter.getContext(), 32), cond);
      std::string icmp = "llvm.mxc.icmp.i64.i32";
      StringRef icmpName(icmp);
      Value cmpModel = b.i32_val(34);
      SmallVector<Value> rangeValueCmp(3);
      rangeValueCmp[0] = maskVal;          // mask to be compared
      rangeValueCmp[1] = b.int_val(32, 0); // compare to 0
      rangeValueCmp[2] = cmpModel;         // compare model
      ValueRange icomValue(rangeValueCmp);
      Value xmask =
          mlir::LLVM::createBuiltinFunc<triton::gpu::AsyncCopyGlobalToLocalOp>(
              rewriter, loc, op, icmpName, i64_ty, icomValue);

      SmallVector<Value> rangeValueLdgBsm(8);
      rangeValueLdgBsm[0] = shmemAddr;       // shared memory addr
      rangeValueLdgBsm[1] = gvmi8Addr;       // gvmbaseptr;
      rangeValueLdgBsm[2] = b.i32_val(0);    // addr offset, default 0
      rangeValueLdgBsm[3] = xmask;           // mask
      rangeValueLdgBsm[4] = b.int_val(1, 1); // enable return0
      rangeValueLdgBsm[5] = b.int_val(1, 1); // flag: whether to use saddr
      rangeValueLdgBsm[6] = b.int_val(1, 0); // pred_neg
      rangeValueLdgBsm[7] = b.int_val(1, 1); // enable async
      ValueRange ldgBsmValue(rangeValueLdgBsm);

      std::string ldgPredicator = "llvm.mxc.ldg.predicator.bsm";
      appendIntrinsicModifer(ldgPredicator, minVec, valueElemTy);
      StringRef ldgBsmFuncName(ldgPredicator);

      // RetType of llvm.mxc.ldg.predicator.f32 should be float
      Type retTy = minVec > 1 ? vec_ty(valueElemTy, minVec) : valueElemTy;
      mlir::LLVM::createBuiltinFunc<triton::gpu::AsyncCopyGlobalToLocalOp>(
          rewriter, loc, op, ldgBsmFuncName, retTy, ldgBsmValue);
      return {};
    };

    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, noSwizzleDstTy);
    auto maskSpanAffineOffset =
        SharedMemoryObject::getMaskSpanOffsets(noSwizzleDstTy);
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    lowerLdSt(
        loc, ctx, cvt, vals, resElemTy, smemObj.getBase(),
        [](Value v) { return v; }, affineOffset, maskSpanAffineOffset, laneId,
        warpId, rewriter, targetInfo, minVec, emitLdgBsm);

    // Drop the result token.
    rewriter.replaceOp(op, llDst);
    return success();
  }
};
} // namespace

void mlir::triton::METAX::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<AsyncCopyGlobalToLocalOpConversion, AtomicCASOpConversion,
               AtomicRMWOpConversion, LoadOpConversion, StoreOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
}
