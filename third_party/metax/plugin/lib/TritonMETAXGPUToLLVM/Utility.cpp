#include "Utility.h"
#include "mlir/Dialect/LLVMIR/MACADialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::gpu {
SmallVector<Value> emitSwizzleColOffsets(RewriterBase &rewriter, Operation *op,
                                         RankedTensorType srcTy,
                                         ArrayRef<int64_t> shape,
                                         Attribute swizzledSharedEnc,
                                         Attribute noSwizzledSharedEnc,
                                         unsigned vec) {
  // LL0: reg -> noswizzle smem offset
  // LL1: reg -> swizzle smem offset
  // colOffset = sharedSwizzledOffset - sharedNoSwizzledOffset
  auto loc = op->getLoc();
  TritonLLVMOpBuilder b(loc, rewriter);
  auto regLayout = triton::gpu::toLinearLayout(srcTy);
  auto swiSmemLayout = triton::gpu::toLinearLayout(shape, swizzledSharedEnc);
  auto noSwiSmemLayout =
      triton::gpu::toLinearLayout(shape, noSwizzledSharedEnc);

  auto regToSwiSmemOffset = regLayout.invertAndCompose(swiSmemLayout);
  auto regToNoSwiSmemOffset = regLayout.invertAndCompose(noSwiSmemLayout);

  MLIRContext *ctx = rewriter.getContext();
  StringAttr kBlock = str_attr("block");
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = b.i32_val(0);

  int numberOfLoads = regToSwiSmemOffset.getInDimSize(kRegister) / vec;
  SmallVector<Value> colOffsets;
  colOffsets.reserve(numberOfLoads);
  auto vecVal = b.i32_val(vec);
  for (int i = 0; i < numberOfLoads; i++) {
    auto regId = b.i32_val(i * vec);
    std::array<std::pair<StringAttr, Value>, 4> indices{{
        {kRegister, regId},
        {kLane, laneId},
        {kWarp, warpId},
        {kBlock, blockId},
    }};

    Value sharedSwizzledOffset =
        applyLinearLayout(loc, rewriter, regToSwiSmemOffset, indices)[0].second;
    Value sharedNoSwizzledOffset =
        applyLinearLayout(loc, rewriter, regToNoSwiSmemOffset, indices)[0]
            .second;

    auto colOffset = b.sub(sharedSwizzledOffset, sharedNoSwizzledOffset);
    colOffsets.push_back(colOffset);
  }
  return colOffsets;
}
} // namespace mlir::triton::gpu

namespace mlir {
namespace LLVM {
namespace METAX {

using namespace mlir::triton;

static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, MACA::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, val0, i, mode, clamp);
    val1 = shuffleCommonImpl(loc, rewriter, val1, i, mode, clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }
  Type type = val.getType();
  if (type != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }
  Value mask = b.i32_val(0xFFFFFFFF);
  Value result = rewriter.create<MACA::ShflOp>(loc, i32_ty, mask, val, i, clamp,
                                               mode, UnitAttr());
  if (type != i32_ty) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result);
    result = b.bitcast(result, type);
  }
  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, MACA::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i, mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), MACA::ShflKind::bfly,
                       b.i32_val(0x1f));
}

Value shuffleXorIntrinsic(Location loc, RewriterBase &rewriter, Value val,
                          int i, Operation *op) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleXorIntrinsic(loc, rewriter, val0, i, op);
    val1 = shuffleXorIntrinsic(loc, rewriter, val1, i, op);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }
  Type type = val.getType();
  if (type != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }

  Value tid = getThreadId(rewriter, loc);
  Value width = b.i32_val(64);       // set width to 64
  Value iValue = b.i32_val(i);       // delta
  Value index = b.xor_(tid, iValue); // tid ^ delta
  Value andValue = b.and_(b.add(tid, width), b.i32_val(-64));
  Value cmpValue = b.icmp_uge(index, andValue); // index >= andValue
  Value selectValue =
      b.select(cmpValue, tid, index); // index>=andValue?tid:index
  Value shflValue = b.shl(selectValue, b.i32_val(2)); // selectValue << 2

  StringRef funcName("llvm.mxc.bsm.bpermute");
  ValueRange valueRange({shflValue, val});
  Type funcType = mlir::triton::gpu::getFunctionType(i32_ty, valueRange);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, op, funcName, funcType);
  auto result =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange).getResult();

  if (type != i32_ty) {
    if (bits < 32) {
      result = b.trunc(int_ty(bits), result);
    }
    result = b.bitcast(result, type);
  }
  return result;
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), MACA::ShflKind::up,
                       b.i32_val(0x0));
}

Value shuffleUpIntrinsic(Location loc, RewriterBase &rewriter, Value val, int i,
                         Operation *op) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();
  auto laneId = getLaneAndWarpId(rewriter, loc).first;
  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleUpIntrinsic(loc, rewriter, val0, i, op);
    val1 = shuffleUpIntrinsic(loc, rewriter, val1, i, op);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  Value warpSize = b.i32_val(64);
  Value self = laneId;
  Value iValue = b.i32_val(i); // laneDelta
  Value index = b.sub(self, iValue);
  Value andValue = b.and_(self, b.i32_val(-64));
  Value cmpValue =
      b.icmp_ult(index, andValue); // (index < (self & ~(width - 1)))
  Value selectValue = b.select(
      cmpValue, self, index); // (index < (self & ~(width - 1))) ? self : index
  Value shflValue = b.shl(selectValue, b.i32_val(2)); // selectValue << 2
  StringRef funcName("llvm.mxc.bsm.bpermute");
  if (bits == 32) {
    Value i32Val = b.bitcast(val, IntegerType::get(rewriter.getContext(), 32));
    ValueRange valueRange({shflValue, i32Val});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, i32Val.getType(),
                                 valueRange);
    return b.bitcast(ret, val.getType());
  } else {
    Value iVal = b.bitcast(val, IntegerType::get(rewriter.getContext(), bits));
    Value zextVal = b.zext(IntegerType::get(rewriter.getContext(), 32), iVal);
    ValueRange valueRange({shflValue, zextVal});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, zextVal.getType(),
                                 valueRange);
    auto truncRet = b.trunc(IntegerType::get(rewriter.getContext(), bits), ret);
    return b.bitcast(truncRet, val.getType());
  }
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, i, MACA::ShflKind::idx,
                       b.i32_val(0x1f));
}

Value shuffleIdxIntrinsic(Location loc, RewriterBase &rewriter, Value val,
                          int i, Operation *op) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdxIntrinsic(loc, rewriter, val, b.i32_val(i), op);
}

Value shuffleIdxIntrinsic(Location loc, RewriterBase &rewriter, Value val,
                          Value i, Operation *op) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleIdxIntrinsic(loc, rewriter, val0, i, op);
    val1 = shuffleIdxIntrinsic(loc, rewriter, val1, i, op);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }
  Value shflValue = b.shl(i, b.i32_val(2)); // selectValue << 2
  StringRef funcName("llvm.mxc.bsm.bpermute");
  if (bits == 32) {
    Value i32Val = b.bitcast(val, IntegerType::get(rewriter.getContext(), 32));
    ValueRange valueRange({shflValue, i32Val});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, i32Val.getType(),
                                 valueRange);
    return b.bitcast(ret, val.getType());
  } else {
    Value iVal = b.bitcast(val, IntegerType::get(rewriter.getContext(), bits));
    Value zextVal = b.zext(IntegerType::get(rewriter.getContext(), 32), iVal);
    ValueRange valueRange({shflValue, zextVal});
    auto ret = createBuiltinFunc(rewriter, loc, op, funcName, zextVal.getType(),
                                 valueRange);
    auto truncRet = b.trunc(IntegerType::get(rewriter.getContext(), bits), ret);
    return b.bitcast(truncRet, val.getType());
  }
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis) {
  assert(moduleOp);
  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  if (numCTAs == 1) {
    Value blockId = ::mlir::gpu::BlockIdOp::create(rewriter, loc,
                                                   mlir::gpu::Dimension(axis));
    return arith::IndexCastOp::create(rewriter, loc, i32_ty, blockId);
  }
  llvm_unreachable("invalid axis");
}

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value selector) {
  llvm_unreachable("unsupported permute codepath");
}

void createSyncWarp(Location loc, OpBuilder &rewriter) {
  llvm_unreachable("unsupported sync warp");
}

LogicalResult lowerLdStMatrix(
    Location loc, LinearLayout cvt, bool transpose,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
    Type llvmElemTy, ConversionPatternRewriter &rewriter,
    const ::triton::METAX::TargetInfo &targetInfo) {
  llvm_unreachable("unsupported lowerLdStMatrix");
  return failure();
}

Type getFunctionType(Type resultType, const ValueRange &operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

LLVM::LLVMFuncOp appendOrGetFuncOp(OpBuilder &rewriter, Operation *op,
                                   StringRef funcName, Type funcType) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(rewriter.getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  mlir::OpBuilder b(op->template getParentOfType<LLVMFuncOp>());
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  return ret;
}

__attribute__((optimize("O0"))) __attribute__((noinline)) Value
createBuiltinFunc(OpBuilder &rewriter, Location loc, Operation *op,
                  StringRef funcName, Type resultType,
                  const ValueRange &argValues) {
  Type funcType = getFunctionType(resultType, argValues);
  auto funcOp = appendOrGetFuncOp(rewriter, op, funcName, funcType);
  return rewriter.create<LLVM::CallOp>(loc, funcOp, argValues).getResult();
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

Value convertBf16ToFp32(Location loc, RewriterBase &rewriter, const Value &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int16 = b.bitcast(v, i16_ty);
  auto as_int32 = b.zext(i32_ty, as_int16);
  auto shifted = b.shl(i32_ty, as_int32, b.i32_val(16));
  return (b.bitcast(shifted, f32_ty));
}

Value convertFp32ToBf16(Location loc, Operation *op, RewriterBase &rewriter,
                        const Value &v,
                        const mlir::triton::RoundingMode rounding) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (rounding == mlir::triton::RoundingMode::RTZ) {
    auto as_int32 = b.bitcast(v, i32_ty);
    auto shifted = b.lshr(i32_ty, as_int32, b.i32_val(16));
    auto truncated = b.trunc(i16_ty, shifted);
    return b.bitcast(truncated, i16_ty);
  } else {
    // use the __float2bfloat16 method instead
    // default way: rounding == RoundingMode::RTNE
    std::string instructionName = "llvm.mxc.cvt.f32tobf16";
    StringRef funcName(instructionName);
    ValueRange valueRange({v});
    Type funcType = mlir::triton::gpu::getFunctionType(i16_ty, valueRange);
    LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType);
    auto ret =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, valueRange).getResult();
    return ret;
  }
}

} // namespace METAX

Type getFunctionType(Type resultType, const ValueRange &operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

} // namespace LLVM
} // namespace mlir
