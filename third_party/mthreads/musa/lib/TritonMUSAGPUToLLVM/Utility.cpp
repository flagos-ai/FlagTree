#include "TritonMUSAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::getFunctionType;
using namespace mlir::triton;

namespace {

static RankedTensorType getSqmmaCarrierTensorType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy;
  if (auto carrierTy = dyn_cast<triton::mtgpu::SqmmaAccumulatorType>(type))
    return carrierTy.getAccumulatorType();
  return RankedTensorType();
}

static FailureOr<LLVM::MUSA::SqmmaAccumulatorCarrierInfo>
buildSqmmaAccumulatorCarrierInfo(Type type) {
  auto tensorTy = getSqmmaCarrierTensorType(type);
  auto mmaEnc =
      tensorTy
          ? dyn_cast<triton::gpu::MUSASqmmaEncodingAttr>(tensorTy.getEncoding())
          : triton::gpu::MUSASqmmaEncodingAttr();
  if (!tensorTy || !mmaEnc || !mmaEnc.isPH1())
    return failure();

  auto instrShape = mmaEnc.getInstrShape();
  auto warpsPerCTA = mmaEnc.getWarpsPerCTA();
  if (instrShape.size() != 3 || warpsPerCTA.size() < 2 || warpsPerCTA[0] < 4)
    return failure();

  auto shapePerCTA = triton::gpu::getShapePerCTA(tensorTy);
  auto ceilDiv = [](unsigned x, unsigned y) {
    return y == 0 ? 0 : (x + y - 1) / y;
  };

  unsigned instM = instrShape[0];
  unsigned instN = instrShape[1];
  unsigned squadsM = warpsPerCTA[0] / 4;
  unsigned squadsN = warpsPerCTA[1];
  unsigned tileM = instM * squadsM;
  unsigned tileN = instN * squadsN;
  unsigned numRepM = std::max(1u, ceilDiv(shapePerCTA[0], tileM));
  unsigned numRepN = std::max(1u, ceilDiv(shapePerCTA[1], tileN));
  unsigned fragmentCount = numRepM * numRepN;
  unsigned totalAccElems = mmaEnc.getTotalElemsPerThread(tensorTy.getShape());
  if (fragmentCount == 0 || totalAccElems == 0 ||
      (totalAccElems % fragmentCount) != 0) {
    return failure();
  }

  MLIRContext *ctx = type.getContext();
  unsigned fragmentElems = totalAccElems / fragmentCount;
  Type fragmentType = VectorType::get({static_cast<int64_t>(fragmentElems)},
                                      tensorTy.getElementType());
  Type carrierType = fragmentType;
  if (fragmentCount > 1) {
    SmallVector<Type> fields(fragmentCount, fragmentType);
    carrierType = LLVM::LLVMStructType::getLiteral(ctx, fields);
  }

  return LLVM::MUSA::SqmmaAccumulatorCarrierInfo{
      tensorTy, fragmentCount, fragmentElems, fragmentType, carrierType};
}

std::string getTypeString(mlir::Type ty) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  ty.print(rso);
  rso.flush();
  return str;
}

std::string mangleFunc(llvm::StringRef name, mlir::Type type) {
  auto funcType = llvm::dyn_cast<mlir::LLVM::LLVMFunctionType>(type);
  assert(funcType && "Expected LLVM function type");
  std::string mangled = name.str();
  mangled.push_back('_');
  mangled += getTypeString(funcType.getReturnType());
  mangled.push_back('_');
  for (auto param : funcType.getParams()) {
    mangled += getTypeString(param);
    mangled.push_back('_');
  }
  return mangled;
}

llvm::StringRef getShuffleIntrinsicName(llvm::StringRef kind) {
  if (kind == "xor")
    return "llvm.musa.shfl.xor.sync.i32";
  if (kind == "up")
    return "llvm.musa.shfl.up.sync.i32";
  if (kind == "down")
    return "llvm.musa.shfl.down.sync.i32";
  return "llvm.musa.shfl.idx.sync.i32";
}

mlir::Value shuffleCommon(mlir::Location loc, mlir::RewriterBase &rewriter,
                          mlir::Value value, mlir::Value offset,
                          llvm::StringRef kind, int widthInt) {
  using namespace mlir;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto valueTy = value.getType();

  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(valueTy)) {
    Value raw = b.ptrtoint(i64_ty, value);
    Value shuffled = shuffleCommon(loc, rewriter, raw, offset, kind, widthInt);
    return b.inttoptr(ptrTy, shuffled);
  }

  unsigned bits = valueTy.getIntOrFloatBitWidth();

  if (bits == 64) {
    auto i64Ty = rewriter.getI64Type();
    Value raw = valueTy.isInteger(64) ? value : b.bitcast(value, i64Ty);
    Value lo = b.trunc(i32_ty, raw);
    Value hi = b.trunc(i32_ty, b.lshr(i64Ty, raw, b.int_val(64, 32)));
    lo = shuffleCommon(loc, rewriter, lo, offset, kind, widthInt);
    hi = shuffleCommon(loc, rewriter, hi, offset, kind, widthInt);
    Value packedLo = b.zext(i64Ty, lo);
    Value packedHi = b.shl(i64Ty, b.zext(i64Ty, hi), b.int_val(64, 32));
    Value packed = b.or_(packedLo, packedHi);
    return valueTy.isInteger(64) ? packed : b.bitcast(packed, valueTy);
  }

  Value val = value;
  if (!valueTy.isInteger(32)) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }

  Value maskAndClamp;
  if (kind == "up") {
    maskAndClamp = b.i32_val(0);
  } else {
    Value width = b.i32_val(widthInt);
    Value clamp = b.sub(width, b.i32_val(1));
    Value segMask = b.sub(b.i32_val(128), width);
    segMask = b.shl(segMask, b.i32_val(7));
    maskAndClamp = b.or_(segMask, clamp);
  }

  auto nullPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 5);
  Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, nullPtrTy);
  auto intrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, getShuffleIntrinsicName(kind), i32_ty,
      {val, offset, maskAndClamp, nullPtr});
  Value result = intrinsic.getResult(0);

  if (!valueTy.isInteger(32)) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result);
    result = b.bitcast(result, valueTy);
  }
  return result;
}

} // namespace

namespace mlir {
namespace LLVM {
namespace MUSA {

FailureOr<SqmmaAccumulatorCarrierInfo>
getSqmmaAccumulatorCarrierInfo(Type type) {
  return buildSqmmaAccumulatorCarrierInfo(type);
}

SmallVector<Value> unpackSqmmaAccumulatorCarrier(Location loc, Value carrier,
                                                 Type type,
                                                 RewriterBase &rewriter) {
  auto info = buildSqmmaAccumulatorCarrierInfo(type);
  assert(succeeded(info) && "expected valid SQMMA accumulator carrier type");
  if (info->fragmentCount == 1)
    return {carrier};

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> fragments;
  fragments.reserve(info->fragmentCount);
  for (unsigned i = 0; i < info->fragmentCount; ++i) {
    fragments.push_back(b.extract_val(info->fragmentType, carrier, i));
  }
  return fragments;
}

Value packSqmmaAccumulatorCarrier(Location loc, ValueRange fragments, Type type,
                                  RewriterBase &rewriter) {
  auto info = buildSqmmaAccumulatorCarrierInfo(type);
  assert(succeeded(info) && "expected valid SQMMA accumulator carrier type");
  assert(fragments.size() == info->fragmentCount &&
         "fragment count mismatch when packing SQMMA carrier");
  if (info->fragmentCount == 1)
    return fragments.front();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value packed = LLVM::UndefOp::create(rewriter, loc, info->carrierType);
  for (unsigned i = 0; i < info->fragmentCount; ++i) {
    packed = b.insert_val(info->carrierType, packed, fragments[i], i);
  }
  return packed;
}

Value carrierFragmentToMathVec(Location loc, Value fragment, Type type,
                               RewriterBase &rewriter) {
  auto info = buildSqmmaAccumulatorCarrierInfo(type);
  assert(succeeded(info) && "expected valid SQMMA accumulator carrier type");
  Type elemTy = info->tensorType.getElementType();
  Type mathVecTy =
      VectorType::get({static_cast<int64_t>(info->fragmentElems)}, elemTy);
  if (fragment.getType() == mathVecTy)
    return fragment;
  return TritonLLVMOpBuilder(loc, rewriter).bitcast(fragment, mathVecTy);
}

Value mathVecToCarrierFragment(Location loc, Value mathVec, Type type,
                               RewriterBase &rewriter) {
  auto info = buildSqmmaAccumulatorCarrierInfo(type);
  assert(succeeded(info) && "expected valid SQMMA accumulator carrier type");
  if (mathVec.getType() == info->fragmentType)
    return mathVec;
  return TritonLLVMOpBuilder(loc, rewriter)
      .bitcast(mathVec, info->fragmentType);
}

Value packSqmmaAccumulatorCarrierFromTensor(Location loc, Value tensorValue,
                                            RankedTensorType tensorType,
                                            const LLVMTypeConverter *converter,
                                            RewriterBase &rewriter) {
  (void)converter;
  auto info = buildSqmmaAccumulatorCarrierInfo(tensorType);
  assert(succeeded(info) && "expected valid SQMMA accumulator tensor");
  SmallVector<Value> elements =
      ::mlir::unpackLLElements(loc, tensorValue, rewriter);
  assert(elements.size() == info->fragmentCount * info->fragmentElems &&
         "unexpected SQMMA accumulator element count");

  SmallVector<Value> fragments;
  fragments.reserve(info->fragmentCount);
  for (unsigned i = 0; i < info->fragmentCount; ++i) {
    ArrayRef<Value> slice(elements);
    Value mathVec = packLLVector(
        loc, slice.slice(i * info->fragmentElems, info->fragmentElems),
        rewriter);
    fragments.push_back(
        mathVecToCarrierFragment(loc, mathVec, tensorType, rewriter));
  }
  return packSqmmaAccumulatorCarrier(loc, fragments, tensorType, rewriter);
}

Value unpackSqmmaAccumulatorCarrierToTensor(Location loc, Value carrier,
                                            RankedTensorType tensorType,
                                            const LLVMTypeConverter *converter,
                                            RewriterBase &rewriter) {
  SmallVector<Value> fragments =
      unpackSqmmaAccumulatorCarrier(loc, carrier, tensorType, rewriter);
  SmallVector<Value> elements;
  auto info = buildSqmmaAccumulatorCarrierInfo(tensorType);
  assert(succeeded(info) && "expected valid SQMMA accumulator tensor");
  elements.reserve(info->fragmentCount * info->fragmentElems);
  for (Value fragment : fragments) {
    Value mathVec =
        carrierFragmentToMathVec(loc, fragment, tensorType, rewriter);
    SmallVector<Value> mathElems = unpackLLVector(loc, mathVec, rewriter);
    elements.append(mathElems.begin(), mathElems.end());
  }
  return ::mlir::packLLElements(loc, converter, elements, rewriter, tensorType);
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), "xor", width);
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                unsigned width) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), "up", width);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 unsigned width) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), "idx", width);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 unsigned width) {
  return shuffleCommon(loc, rewriter, val, i, "idx", width);
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp /*moduleOp*/,
               triton::ProgramIDDim axis) {
  StringRef intrinsic;
  switch (axis) {
  case triton::ProgramIDDim::X:
    intrinsic = "llvm.musa.read.ptx.sreg.ctaid.x";
    break;
  case triton::ProgramIDDim::Y:
    intrinsic = "llvm.musa.read.ptx.sreg.ctaid.y";
    break;
  case triton::ProgramIDDim::Z:
    intrinsic = "llvm.musa.read.ptx.sreg.ctaid.z";
    break;
  default:
    intrinsic = "llvm.musa.read.ptx.sreg.ctaid.x";
    break;
  }

  auto call =
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty, {});
  return call.getResult(0);
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal) {
  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(Predicated_Load, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  auto loadVal = LLVM::CallOp::create(rewriter, loc, funcOp,
                                      ValueRange({ptr, pred, falseVal}))
                     .getResult();
  return loadVal;
}

Value llInplaceLoad(RewriterBase &rewriter, Location loc, Value ptr,
                    Type elemTy, Value pred, Value falseVal) {
  Type funcType = getFunctionType(elemTy, ValueRange({ptr, pred, falseVal}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(Predicated_InplaceLoad, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  auto loadVal = LLVM::CallOp::create(rewriter, loc, funcOp,
                                      ValueRange({ptr, pred, falseVal}))
                     .getResult();
  return loadVal;
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred) {
  Type funcType = getFunctionType(void_ty(rewriter.getContext()),
                                  ValueRange({ptr, val, pred}));
  auto parent = ptr.getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>();
  auto funcName = mangleFunc(Predicated_Store, funcType);
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, parent, funcName, funcType);
  LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange({ptr, val, pred}));
}

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value mask) {
  auto bld = TritonLLVMOpBuilder(loc, rewriter);
  Type valueTy = a.getType();
  assert(valueTy == b.getType() && "permute operands must have the same type");

  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(valueTy)) {
    Value aI64 = bld.ptrtoint(i64_ty, a);
    Value bI64 = bld.ptrtoint(i64_ty, b);
    Value resultI64 = permute(loc, rewriter, aI64, bI64, mask);
    return bld.inttoptr(ptrTy, resultI64);
  }

  unsigned bits = valueTy.getIntOrFloatBitWidth();
  if (bits == 64) {
    Value aI64 = valueTy.isInteger(64) ? a : bld.bitcast(a, i64_ty);
    Value bI64 = valueTy.isInteger(64) ? b : bld.bitcast(b, i64_ty);

    Value aLo = bld.trunc(i32_ty, aI64);
    Value aHi = bld.trunc(i32_ty, bld.lshr(i64_ty, aI64, bld.i64_val(32)));
    Value bLo = bld.trunc(i32_ty, bI64);
    Value bHi = bld.trunc(i32_ty, bld.lshr(i64_ty, bI64, bld.i64_val(32)));

    Value outLo = permute(loc, rewriter, aLo, bLo, mask);
    Value outHi = permute(loc, rewriter, aHi, bHi, mask);
    Value packedLo = bld.zext(i64_ty, outLo);
    Value packedHi = bld.shl(i64_ty, bld.zext(i64_ty, outHi), bld.i64_val(32));
    Value packed = bld.or_(packedLo, packedHi);
    return valueTy.isInteger(64) ? packed : bld.bitcast(packed, valueTy);
  }

  Type rawTy = valueTy.isIntOrIndex() ? valueTy : int_ty(bits);
  Value aI32 = a;
  Value bI32 = b;
  if (!valueTy.isIntOrIndex()) {
    aI32 = bld.bitcast(a, rawTy);
    bI32 = bld.bitcast(b, rawTy);
  }
  if (bits < 32) {
    aI32 = bld.zext(i32_ty, aI32);
    bI32 = bld.zext(i32_ty, bI32);
  } else if (bits != 32) {
    llvm_unreachable("permute only supports scalar values up to 64 bits");
  } else if (!valueTy.isInteger(32)) {
    aI32 = bld.bitcast(aI32, i32_ty);
    bI32 = bld.bitcast(bI32, i32_ty);
  }

  auto call = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.musa.prmt",
                                              i32_ty, {aI32, bI32, mask});
  Value result = call.getResult(0);
  if (bits < 32)
    result = bld.trunc(rawTy, result);
  if (!valueTy.isIntOrIndex())
    result = bld.bitcast(result, valueTy);
  return result;
}

Value createElectPredicate(Location loc, PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value laneId =
      LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, "llvm.musa.read.ptx.sreg.laneid", i32_ty, {})
          .getResult(0);
  Value firstLane =
      LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, "llvm.musa.vote.firstactivelane", i32_ty, {})
          .getResult(0);
  return b.icmp_eq(laneId, firstLane);
}

LLVM::LLVMFuncOp getLibdeviceFuncCall(RewriterBase &rewriter, Operation *op,
                                      StringRef funcName, Type retType,
                                      ValueRange ins) {
  Type funcType = getFunctionType(retType, ins);
  return appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
}

} // namespace MUSA
} // namespace LLVM
} // namespace mlir
