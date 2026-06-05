/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "../MACACommonConversion.h"
#include "Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::LLVM;
using namespace mlir::triton::gpu;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using MACAMmaEncodingAttr = ::mlir::triton::gpu::MACAMmaEncodingAttr;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;

namespace mlir {
namespace MACA {

// Get the M and N of mma instruction shape.
static std::tuple<int, int> getInstrShapeMN() {
  // According to MACA mma instructions, all the M,N are {16,16}
  return {16, 16};
}

static const std::map<TensorCoreType, std::string> mmaInstrPtx = {
    {TensorCoreType::FP32_FP16_FP16_FP32, "mma_16x16x16f16"},
    {TensorCoreType::FP32_BF16_BF16_FP32, "mma_16x16x16bf16"},
    {TensorCoreType::FP32_FP32_FP32_FP32, "mma_16x16x4f32"},
    {TensorCoreType::FP32_TF32_TF32_FP32, "mma_16x16x8tf32"},
    {TensorCoreType::FP64_FP64_FP64_FP64, "mma_16x16x4f64"},
    {TensorCoreType::INT32_INT8_INT8_INT32, "mma_16x16x16i8"},
    {TensorCoreType::INT32_INT8_INT8_INT32_1, "mma_16x16x32i8"},
    {TensorCoreType::FP32_FP8_FP8_FP32, "mma_16x16x32f8"},
    {TensorCoreType::FP32_BF8_BF8_FP32, "mma_16x16x32bf8"},
};

static const std::map<std::string, std::string> mmaInstrBuiltin = {
    {"mma_16x16x16f16", "llvm.mxc.mma.f32.16x16x16f16"},
    {"mma_16x16x16bf16", "llvm.mxc.mma.f32.16x16x2bf16"},
    {"mma_16x16x4f32", "llvm.mxc.mma.f32.16x16x4f32"},
    {"mma_16x16x8tf32", "llvm.mxc.mma.f32.16x16x8tf32"},
    {"mma_16x16x4f64", "llvm.mxc.mma.f64.16x16x4f64"},
    {"mma_16x16x16i8", "llvm.mxc.mma.i32.16x16x16i8"},
    {"mma_16x16x32i8", "llvm.mxc.mma.i32.16x16x32i8"},
    {"mma_16x16x32f8", "llvm.mxc.mma.f32.16x16x32f8"},
    {"mma_16x16x32bf8", "llvm.mxc.mma.f32.16x16x32bf8"},
};

static std::tuple<int, int> getRepMN(const RankedTensorType &tensorTy) {
  auto mmaLayout = cast<MACAMmaEncodingAttr>(tensorTy.getEncoding());
  auto wpt = mmaLayout.getWarpsPerCTA();
  auto elemsMnk = mmaLayout.getElementsMNK();

  int M = tensorTy.getShape()[0];
  int N = tensorTy.getShape()[1];
  auto [instrM, instrN] = getInstrShapeMN();
  int repM = std::max<int>(M / (wpt[0] * instrM * elemsMnk[0]), 1);
  int repN = std::max<int>(N / (wpt[1] * instrN * elemsMnk[1]), 1);
  return {repM, repN};
}

// get the number of elements of each mma C/D matrix
static int getMmaRetNum(TensorCoreType mmaType) {
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return 4;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return 4;
  case TensorCoreType::FP32_FP32_FP32_FP32:
    return 4;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return 4;
  case TensorCoreType::FP64_FP64_FP64_FP64:
    return 4;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return 4;
  case TensorCoreType::INT32_INT8_INT8_INT32_1:
    return 4;
  case TensorCoreType::FP32_FP8_FP8_FP32:
    return 4;
  case TensorCoreType::FP32_BF8_BF8_FP32:
    return 4;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }
  return 0;
}

// get the number of elements of each mma A/B matrix
static int getMmaOperandNum(TensorCoreType mmaType) {
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return 4;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return 4;
  case TensorCoreType::FP32_FP32_FP32_FP32:
    return 1;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return 2;
  case TensorCoreType::FP64_FP64_FP64_FP64:
    return 1;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return 4;
  case TensorCoreType::INT32_INT8_INT8_INT32_1:
    return 8;
  case TensorCoreType::FP32_FP8_FP8_FP32:
    return 8;
  case TensorCoreType::FP32_BF8_BF8_FP32:
    return 8;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return 0;
}

StringRef getMmaInstr(TensorCoreType mmaType) {
  assert(mmaType != TensorCoreType::NOT_APPLICABLE &&
         "Unknown mma type found.");
  return mmaInstrPtx.at(mmaType);
}

StringRef getMmaInstrBuiltin(std::string mmaInstr) {
  return StringRef(mmaInstrBuiltin.at(mmaInstr));
}

static TensorCoreType getMmaType(triton::DotOp op, int major, int minor) {
  Value A = op.getA();
  Value B = op.getB();
  auto aTy = cast<RankedTensorType>(A.getType());
  auto bTy = cast<RankedTensorType>(B.getType());
  // d = a*b + c
  auto dTy = cast<RankedTensorType>(op.getD().getType());

  if (dTy.getElementType().isF32()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP32_FP16_FP16_FP32;
    if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
      return TensorCoreType::FP32_BF16_BF16_FP32;

    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        !(op.getInputPrecision() == InputPrecision::TF32))
      return TensorCoreType::FP32_FP32_FP32_FP32;
    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        (op.getInputPrecision() == InputPrecision::TF32)) {
      return TensorCoreType::FP32_TF32_TF32_FP32;
    }
    if (llvm::isa<Float8E5M2Type>(aTy.getElementType()) &&
        llvm::isa<Float8E5M2Type>(bTy.getElementType())) {
      return TensorCoreType::FP32_BF8_BF8_FP32;
    }
    if (llvm::isa<Float8E4M3FNType>(aTy.getElementType()) &&
        llvm::isa<Float8E4M3FNType>(bTy.getElementType())) {
      return TensorCoreType::FP32_FP8_FP8_FP32;
    }
  }

  if (dTy.getElementType().isF64()) {
    if (aTy.getElementType().isF64() && bTy.getElementType().isF64())
      return TensorCoreType::FP64_FP64_FP64_FP64;
  }

  if (dTy.getElementType().isSignlessInteger(32)) {
    if (aTy.getElementType().isSignlessInteger(8) &&
        bTy.getElementType().isSignlessInteger(8))
      if (major == 2 && minor >= 6) {
        return TensorCoreType::INT32_INT8_INT8_INT32_1;
      } else {
        return TensorCoreType::INT32_INT8_INT8_INT32;
      }
  }

  return TensorCoreType::NOT_APPLICABLE;
}

// Loading $c to registers, returns a Value.
Value loadC(Value tensor, Value llTensor,
            const LLVMTypeConverter *typeConverter, Location loc,
            ConversionPatternRewriter &rewriter) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto mmaLayout = cast<MACAMmaEncodingAttr>(tensorTy.getEncoding());
  auto elemsMnk = mmaLayout.getElementsMNK();
  auto [repM, repN] = getRepMN(tensorTy);
  // each thread get 4 elements each mma for C matrix;
  size_t fcSize = 4 * repM * repN * elemsMnk[0] * elemsMnk[1];

  assert(isa<MACAMmaEncodingAttr>(tensorTy.getEncoding()) &&
         "Currently, we only support $c with a mma layout.");
  // Load a normal C tensor with mma layout, that should be a
  // LLVM::struct with fcSize elements.
  auto structTy = cast<LLVM::LLVMStructType>(llTensor.getType());
  assert(structTy.getBody().size() == fcSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");
  return llTensor;
}

ValueTable getValuesFromDotOperandLayoutStruct(
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Value value, int n0, int n1,
    RankedTensorType type, int elemsPerThreadMN, int elemsPerThreadK) {

  auto elems = unpackLLElements(loc, value, rewriter);
  int offset{};
  ValueTable vals;
  for (int m = 0; m < n0; ++m)
    for (int k = 0; k < n1; ++k)
      for (int j = 0; j < elemsPerThreadMN; ++j)
        for (int i = 0; i < elemsPerThreadK; ++i) {
          vals[{m * elemsPerThreadMN + j, k * elemsPerThreadK + i}] =
              elems[offset++];
        }
  return vals;
}

// Conduct the Dot conversion.
LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value a, Value b, Value c, Value d, Value loadedA,
                         Value loadedB, Value loadedC, DotOp op,
                         DotOpAdaptor adaptor) {
  MLIRContext *ctx = c.getContext();
  auto builder = TritonLLVMOpBuilder(loc, rewriter);

  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto cTensorTy = cast<RankedTensorType>(c.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  SmallVector<int64_t> aShape(aTensorTy.getShape().begin(),
                              aTensorTy.getShape().end());
  auto bShape = bTensorTy.getShape();
  llvm::SmallVector<int64_t, 3> tile({aShape[0], bShape[1], aShape[1]});
  auto mmaLayout = cast<MACAMmaEncodingAttr>(cTensorTy.getEncoding());
  auto major = mmaLayout.getVersionMajor();
  auto minor = mmaLayout.getVersionMinor();
  auto mmaType = getMmaType(op, major, minor);
  auto wpt = mmaLayout.getWarpsPerCTA();
  auto elemsMnk = mmaLayout.getElementsMNK();
  auto dShape = dTensorTy.getShape();
  Type inElemTy = aTensorTy.getElementType();
  bool isColMajor = mmaLayout.getColMajor();
  // mma output colmajor only support fp16/bf16 for now
  if (isColMajor) {
    assert((inElemTy.isF16() || inElemTy.isBF16()) &&
           "MMA Output colmajor now only support fp16/bf16");
  }

  // elems loaded of each thread per replica for M.
  int elemsM = elemsMnk[0];
  // elems loaded of each thread per replica for N.
  int elemsN = elemsMnk[1];
  // elems loaded of each thread per replica for K.
  int elemsK = elemsMnk[2];

  // shape / shape_per_cta
  int numRepM = getNumRepM(dShape[0], wpt, elemsM);
  int numRepN = getNumRepN(dShape[1], wpt, elemsN);
  int numRepK = getNumRepK(aShape[1], elemsK);
  auto ha = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                loadedA, numRepM, numRepK,
                                                aTensorTy, elemsM, elemsK);
  auto hb = getValuesFromDotOperandLayoutStruct(
      typeConverter, loc, rewriter, loadedB, std::max(numRepN, 1), numRepK,
      bTensorTy, elemsN, elemsK);
  // auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter,
  // dTensorTy);
  auto fc = unpackLLElements(loc, loadedC, rewriter);
  int retElemsPerMMA = getMmaRetNum(mmaType);
  int operandElemsPerMMA = getMmaOperandNum(mmaType);

  auto callF32Mma = [&](unsigned m, unsigned n, unsigned k, unsigned a,
                        unsigned b, unsigned c) {
    // for mma which return FP32 data type, each thread gets 4 elements for C
    // matrix.
    unsigned elementPerThread = 4;
    unsigned colsPerThread = numRepN * elementPerThread;
    auto mma_instr = getMmaInstr(mmaType).str();

    auto v4_t = vec_ty(f32_ty, 4);
    llvm::SmallVector<Value, 3> args_value;
    if (mma_instr == "mma_16x16x16bf16") {
      assert(elemsM == 1 && "load n matrix only support fp32.");
      assert(elemsN == 1 && "load n matrix only support fp32.");
      // for mma_16x16x16bf16 need convert <4 x i16> to <4 x half> in
      // llvm.mxc.mma.f32.16x16x2bf16
      auto convertI16ToHalf = [&](Value a) -> Value {
        auto v4_ta = vec_ty(f16_ty, 4);
        Value v4_va = builder.undef(v4_ta);
        for (int i = 0; i < 4; ++i) {
          Value ta_value = builder.extract_element(
              IntegerType::get(rewriter.getContext(), 16), a,
              builder.i32_val(i));
          Value cast_ta_value = builder.bitcast(ta_value, f16_ty);
          v4_va = builder.insert_element(v4_ta, v4_va, cast_ta_value,
                                         builder.i32_val(i));
        }
        return v4_va;
      };
      args_value.push_back(convertI16ToHalf(ha[{m, k}]));
      args_value.push_back(convertI16ToHalf(hb[{n, k}]));
    } else {
      args_value.push_back(ha[{m * elemsM + a, k * elemsK + c}]); // a
      args_value.push_back(hb[{n * elemsN + b, k * elemsK + c}]); // b
    }
    Value v4_v = builder.undef(v4_t);
    for (size_t ii = 0; ii < 4; ++ii) {
      v4_v = builder.insert_element(
          v4_t, v4_v,
          fc[(m * colsPerThread + elementPerThread * n + ii) * elemsM * elemsN +
             a * elemsN + b],
          builder.i32_val(ii));
    }
    args_value.push_back(v4_v); // c
    ValueRange valueRange(args_value);
    Value v4_value = createBuiltinFunc<DotOp>(rewriter, loc, op,
                                              getMmaInstrBuiltin(mma_instr),
                                              v4_v.getType(), valueRange);
    for (int i = 0; i < 4; ++i) {
      Value v1_value =
          builder.extract_element(f32_ty, v4_value, builder.i32_val(i));
      fc[(m * colsPerThread + elementPerThread * n + i) * elemsM * elemsN +
         a * elemsN + b] = v1_value;
    }
  };

  auto callTF32Mma = [&](unsigned m, unsigned n, unsigned k, unsigned a,
                         unsigned b, unsigned c) {
    // for mma which return FP32 data type, each thread gets 4 elements for C
    // matrix.
    unsigned elementPerThread = 4;
    unsigned colsPerThread = numRepN * elementPerThread;
    auto mma_instr = getMmaInstr(mmaType).str();

    auto v4_t = vec_ty(f32_ty, 4);

    llvm::SmallVector<Value, 3> args_value;
    auto fp32x2Ty = vec_ty(f32_ty, 2);
    Value aElemX2 = builder.undef(fp32x2Ty);
    for (size_t i = 0; i < 2; i++) {
      aElemX2 = builder.insert_element(
          fp32x2Ty, aElemX2, ha[{m * elemsM + a, k * elemsK + c * 2 + i}],
          builder.i32_val(i));
    }
    Value bElemX2 = builder.undef(fp32x2Ty);
    for (size_t i = 0; i < 2; i++) {
      bElemX2 = builder.insert_element(
          fp32x2Ty, bElemX2, hb[{n * elemsN + b, k * elemsK + c * 2 + i}],
          builder.i32_val(i));
    }
    args_value.push_back(aElemX2); // a
    args_value.push_back(bElemX2); // b
    Value v4_v = builder.undef(v4_t);
    for (size_t ii = 0; ii < 4; ++ii) {
      v4_v = builder.insert_element(
          v4_t, v4_v,
          fc[(m * colsPerThread + elementPerThread * n + ii) * elemsM * elemsN +
             a * elemsN + b],
          builder.i32_val(ii));
    }
    args_value.push_back(v4_v); // c
    ValueRange valueRange(args_value);
    Value v4_value = createBuiltinFunc<DotOp>(rewriter, loc, op,
                                              getMmaInstrBuiltin(mma_instr),
                                              v4_v.getType(), valueRange);
    for (int i = 0; i < 4; ++i) {
      Value v1_value =
          builder.extract_element(f32_ty, v4_value, builder.i32_val(i));
      fc[(m * colsPerThread + elementPerThread * n + i) * elemsM * elemsN +
         a * elemsN + b] = v1_value;
    }
  };

  auto callF16Mma = [&](unsigned m, unsigned n, unsigned k, unsigned a,
                        unsigned b, unsigned c) {
    // for mma which return FP32 data type, each thread gets 4 elements for C
    // matrix.
    unsigned colsPerThread = numRepN * retElemsPerMMA;
    auto mma_instr = getMmaInstr(mmaType).str();

    auto v4_t = vec_ty(f32_ty, retElemsPerMMA);
    auto a4_t = vec_ty(f16_ty, operandElemsPerMMA);
    auto b4_t = vec_ty(f16_ty, operandElemsPerMMA);
    Value a4_v = builder.undef(a4_t);
    Value b4_v = builder.undef(b4_t);
    for (size_t ii = 0; ii < operandElemsPerMMA; ++ii) {
      Value a_v =
          ha[{m * elemsM + a, k * elemsK + c * operandElemsPerMMA + ii}];
      Value b_v =
          hb[{n * elemsN + b, k * elemsK + c * operandElemsPerMMA + ii}];
      if (inElemTy.isBF16()) {
        a_v = builder.bitcast(a_v, f16_ty);
        b_v = builder.bitcast(b_v, f16_ty);
      }
      a4_v = builder.insert_element(a4_t, a4_v, a_v, builder.i32_val(ii));
      b4_v = builder.insert_element(b4_t, b4_v, b_v, builder.i32_val(ii));
    }
    llvm::SmallVector<Value, 3> args_value;
    if (isColMajor) {
      args_value.push_back(b4_v); // b4_v
      args_value.push_back(a4_v); // a4_v
    } else {
      args_value.push_back(a4_v); // a4_v
      args_value.push_back(b4_v); // b4_v
    }
    Value v4_v = builder.undef(v4_t);
    for (size_t ii = 0; ii < retElemsPerMMA; ++ii) {
      if (isColMajor) {
        v4_v = builder.insert_element(
            v4_t, v4_v,
            fc[m * colsPerThread * elemsM * elemsN +
               retElemsPerMMA * n * elemsN * elemsM + ii * elemsN +
               a * elemsN * retElemsPerMMA + b],
            builder.i32_val(ii));
      } else {
        v4_v = builder.insert_element(
            v4_t, v4_v,
            fc[(m * colsPerThread + retElemsPerMMA * n + ii) * elemsM * elemsN +
               a * elemsN + b],
            builder.i32_val(ii));
      }
    }
    args_value.push_back(v4_v); // c
    ValueRange valueRange(args_value);
    Value v4_value = createBuiltinFunc<DotOp>(rewriter, loc, op,
                                              getMmaInstrBuiltin(mma_instr),
                                              v4_v.getType(), valueRange);
    for (int i = 0; i < retElemsPerMMA; ++i) {
      Value v1_value =
          builder.extract_element(f32_ty, v4_value, builder.i32_val(i));
      if (isColMajor) {
        fc[m * colsPerThread * elemsM * elemsN +
           retElemsPerMMA * n * elemsN * elemsM + i * elemsN +
           a * elemsN * retElemsPerMMA + b] = v1_value;
      } else {
        fc[(m * colsPerThread + retElemsPerMMA * n + i) * elemsM * elemsN +
           a * elemsN + b] = v1_value;
      }
    }
  };

  auto callF64Mma = [&](unsigned m, unsigned n, unsigned k) {
    // for mma which return FP64 data type, each thread gets 4 elements for C
    // matrix.
    unsigned elementPerThread = 4;
    unsigned colsPerThread = numRepN * elementPerThread;
    auto mma_instr = getMmaInstr(mmaType).str();

    auto v4_t = vec_ty(f64_ty, 4);
    llvm::SmallVector<Value, 3> args_value;
    args_value.push_back(ha[{m, k}]); // a
    args_value.push_back(hb[{n, k}]); // b
    Value v4_v = builder.undef(v4_t);
    for (size_t ii = 0; ii < 4; ++ii) {
      v4_v = builder.insert_element(
          v4_t, v4_v, fc[m * colsPerThread + elementPerThread * n + ii],
          builder.i32_val(ii));
    }
    args_value.push_back(v4_v); // c
    ValueRange valueRange(args_value);
    Value v4_value = createBuiltinFunc<DotOp>(rewriter, loc, op,
                                              getMmaInstrBuiltin(mma_instr),
                                              v4_v.getType(), valueRange);
    for (int i = 0; i < 4; ++i) {
      Value v1_value =
          builder.extract_element(f64_ty, v4_value, builder.i32_val(i));
      fc[m * colsPerThread + elementPerThread * n + i] = v1_value;
    }
  };

  auto callB8Mma = [&](unsigned m, unsigned n, unsigned k, unsigned a,
                       unsigned b, unsigned c, Type operandTy, Type resTy) {
    // for mma which return INT32 data type, each thread gets 4 elements for C
    // matrix.
    unsigned colsPerThread = numRepN * retElemsPerMMA;
    auto mma_instr = getMmaInstr(mmaType).str();

    auto v4_t = vec_ty(resTy, retElemsPerMMA);
    auto a4_t = vec_ty(operandTy, operandElemsPerMMA);
    auto b4_t = vec_ty(operandTy, operandElemsPerMMA);
    Value a4_v = builder.undef(a4_t);
    Value b4_v = builder.undef(b4_t);
    for (size_t ii = 0; ii < operandElemsPerMMA; ++ii) {
      Value a_v =
          ha[{m * elemsM + a, k * elemsK + c * operandElemsPerMMA + ii}];
      Value b_v =
          hb[{n * elemsN + b, k * elemsK + c * operandElemsPerMMA + ii}];
      a4_v = builder.insert_element(a4_t, a4_v, a_v, builder.i32_val(ii));
      b4_v = builder.insert_element(b4_t, b4_v, b_v, builder.i32_val(ii));
    }
    Value cast_ha_value, cast_hb_value;
    llvm::SmallVector<Value, 3> args_value;
    if (operandElemsPerMMA / 4 == 1) {
      cast_ha_value = builder.bitcast(a4_v, i32_ty);
      cast_hb_value = builder.bitcast(b4_v, i32_ty);
    } else {
      Type operandType = vec_ty(i32_ty, operandElemsPerMMA / 4);
      cast_ha_value = builder.bitcast(a4_v, operandType);
      cast_hb_value = builder.bitcast(b4_v, operandType);
    }
    args_value.push_back(cast_ha_value); // a4_v
    args_value.push_back(cast_hb_value); // b4_v

    Value v4_v = builder.undef(v4_t);
    for (size_t ii = 0; ii < retElemsPerMMA; ++ii) {
      v4_v = builder.insert_element(
          v4_t, v4_v,
          fc[(m * colsPerThread + retElemsPerMMA * n + ii) * elemsM * elemsN +
             a * elemsN + b],
          builder.i32_val(ii));
    }
    args_value.push_back(v4_v); // c
    ValueRange valueRange(args_value);
    Value v4_value = createBuiltinFunc<DotOp>(rewriter, loc, op,
                                              getMmaInstrBuiltin(mma_instr),
                                              v4_v.getType(), valueRange);
    for (int i = 0; i < retElemsPerMMA; ++i) {
      Value v1_value =
          builder.extract_element(resTy, v4_value, builder.i32_val(i));
      fc[(m * colsPerThread + retElemsPerMMA * n + i) * elemsM * elemsN +
         a * elemsN + b] = v1_value;
    }
  };

  Type resElemTy = dTensorTy.getElementType();
  int elemsK_ = elemsK / operandElemsPerMMA;
  assert(elemsK_ >= 1);

  for (int k = 0; k < numRepK; ++k)
    for (int m = 0; m < numRepM; ++m)
      for (int n = 0; n < numRepN; ++n)
        for (int a = 0; a < elemsM; ++a)
          for (int b = 0; b < elemsN; ++b)
            for (int c = 0; c < elemsK_; ++c)
              if (resElemTy.isF32()) {
                if (inElemTy.isF16() || inElemTy.isBF16()) {
                  callF16Mma(m, n, k, a, b, c);
                } else if (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(
                               inElemTy)) {
                  callB8Mma(m, n, k, a, b, c, i8_ty, resElemTy);
                } else if (op.getInputPrecision() == InputPrecision::TF32) {
                  callTF32Mma(m, n, k, a, b, c);
                } else {
                  callF32Mma(m, n, k, a, b, c);
                }
              } else if (resElemTy.isF64()) {
                callF64Mma(m, n, k);
              } else if ((resElemTy.isSignlessInteger(32) &&
                          inElemTy.isSignlessInteger(8))) {
                callB8Mma(m, n, k, a, b, c, inElemTy, resElemTy);
              } else {
                assert(false && "Only support FP32 mma!");
              }

  for (auto &elem : fc) {
    elem = builder.bitcast(elem, resElemTy);
  }

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(fc.size(), resElemTy));

  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);
  rewriter.replaceOp(op, res);

  return success();
}
} // namespace MACA
} // namespace mlir

using namespace mlir::MACA;

// Convert to maca mma
LogicalResult convertMMAMACA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto mmaLayout = cast<MACAMmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  Value A = op.getA();
  Value B = op.getB();
  Value C = op.getC();

  auto ATensorTy = cast<RankedTensorType>(A.getType());
  auto BTensorTy = cast<RankedTensorType>(B.getType());

  assert(isa<DotOperandEncodingAttr>(ATensorTy.getEncoding()) &&
         isa<DotOperandEncodingAttr>(BTensorTy.getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  Value loadedA, loadedB, loadedC;
  loadedA = adaptor.getA();
  loadedB = adaptor.getB();
  loadedC =
      loadC(op.getC(), adaptor.getC(), typeConverter, op.getLoc(), rewriter);
  return convertDot(typeConverter, rewriter, op.getLoc(), A, B, C, op.getD(),
                    loadedA, loadedB, loadedC, op, adaptor);
}
