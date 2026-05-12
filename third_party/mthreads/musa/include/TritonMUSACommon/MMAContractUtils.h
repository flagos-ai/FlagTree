#ifndef TRITONMUSA_COMMON_MMA_CONTRACT_UTILS_H
#define TRITONMUSA_COMMON_MMA_CONTRACT_UTILS_H

#include "Dialect/MUSA/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::triton::musa {

inline bool isFloat8E4M3(Type ty) {
  return llvm::isa<Float8E4M3FNType, Float8E4M3FNUZType>(ty);
}

inline bool isFloat8E5M2(Type ty) {
  return llvm::isa<Float8E5M2Type, Float8E5M2FNUZType>(ty);
}

inline std::optional<SQMMAEltType> getWmmaEltType(Type elemTy) {
  if (elemTy.isInteger(8))
    return SQMMAEltType::s8;
  if (elemTy.isF16())
    return SQMMAEltType::f16;
  if (elemTy.isBF16())
    return SQMMAEltType::bf16;
  if (elemTy.isF32())
    return SQMMAEltType::tf32;
  if (isFloat8E4M3(elemTy))
    return SQMMAEltType::e4m3;
  if (isFloat8E5M2(elemTy))
    return SQMMAEltType::e5m2;
  return std::nullopt;
}

struct WmmaIntrinsicSignature {
  SQMMAEltType eltType;
  unsigned m;
  unsigned n;
  unsigned k;
  const char *name;
  int32_t sat;
};

inline constexpr WmmaIntrinsicSignature kWmmaIntrinsics[] = {
    {SQMMAEltType::s8, 8, 16, 16, "llvm.musa.imma.m8n16k16.mma", 1},
    {SQMMAEltType::s8, 16, 8, 16, "llvm.musa.imma.m16n8k16.mma", 1},
    {SQMMAEltType::s8, 16, 16, 16, "llvm.musa.imma.m16n16k16.mma", 1},
    {SQMMAEltType::s8, 16, 16, 32, "llvm.musa.imma.m16n16k32.mma", 1},
    {SQMMAEltType::s8, 16, 16, 64, "llvm.musa.imma.m16n16k64.mma", 1},
    {SQMMAEltType::f16, 16, 8, 8, "llvm.musa.ffmma.m16n8k8.mma", 1},
    {SQMMAEltType::f16, 16, 8, 16, "llvm.musa.ffmma.m16n8k16.mma", 1},
    {SQMMAEltType::f16, 8, 16, 16, "llvm.musa.ffmma.m8n16k16.mma", 1},
    {SQMMAEltType::f16, 16, 16, 16, "llvm.musa.ffmma.m16n16k16.mma", 1},
    {SQMMAEltType::f16, 16, 16, 32, "llvm.musa.ffmma.m16n16k32.mma", 1},
    {SQMMAEltType::bf16, 16, 8, 8, "llvm.musa.bfmma.m16n8k8.mma", 1},
    {SQMMAEltType::bf16, 16, 8, 16, "llvm.musa.bfmma.m16n8k16.mma", 1},
    {SQMMAEltType::bf16, 8, 16, 16, "llvm.musa.bfmma.m8n16k16.mma", 1},
    {SQMMAEltType::bf16, 16, 16, 16, "llvm.musa.bfmma.m16n16k16.mma", 1},
    {SQMMAEltType::bf16, 16, 16, 32, "llvm.musa.bfmma.m16n16k32.mma", 1},
    {SQMMAEltType::tf32, 16, 8, 4, "llvm.musa.tfmma.m16n8k4.mma", 1},
    {SQMMAEltType::tf32, 16, 8, 8, "llvm.musa.tfmma.m16n8k8.mma", 1},
    {SQMMAEltType::tf32, 16, 16, 16, "llvm.musa.tfmma.m16n16k16.mma", 1},
    {SQMMAEltType::e4m3, 8, 16, 16, "llvm.musa.e4m3.m8n16k16.mma", 1},
    {SQMMAEltType::e4m3, 16, 8, 16, "llvm.musa.e4m3.m16n8k16.mma", 1},
    {SQMMAEltType::e4m3, 16, 16, 16, "llvm.musa.e4m3.m16n16k16.mma", 1},
    {SQMMAEltType::e4m3, 16, 16, 32, "llvm.musa.e4m3.m16n16k32.mma", 1},
    {SQMMAEltType::e4m3, 16, 16, 64, "llvm.musa.e4m3.m16n16k64.mma", 1},
    {SQMMAEltType::e5m2, 8, 16, 16, "llvm.musa.e5m2.m8n16k16.mma", 1},
    {SQMMAEltType::e5m2, 16, 8, 16, "llvm.musa.e5m2.m16n8k16.mma", 1},
    {SQMMAEltType::e5m2, 16, 16, 16, "llvm.musa.e5m2.m16n16k16.mma", 1},
    {SQMMAEltType::e5m2, 16, 16, 32, "llvm.musa.e5m2.m16n16k32.mma", 1},
    {SQMMAEltType::e5m2, 16, 16, 64, "llvm.musa.e5m2.m16n16k64.mma", 1},
};

inline std::optional<llvm::StringRef>
lookupWmmaIntrinsicName(SQMMAEltType eltType, unsigned m, unsigned n,
                        unsigned k) {
  for (const auto &def : kWmmaIntrinsics) {
    if (def.eltType == eltType && def.m == m && def.n == n && def.k == k)
      return llvm::StringRef(def.name);
  }
  return std::nullopt;
}

inline std::optional<WmmaIntrinsicSignature>
lookupWmmaIntrinsic(Type elemTy, ArrayRef<unsigned> instrShape) {
  if (instrShape.size() != 3)
    return std::nullopt;
  auto eltType = getWmmaEltType(elemTy);
  if (!eltType)
    return std::nullopt;
  for (const auto &def : kWmmaIntrinsics) {
    if (def.eltType == *eltType && def.m == instrShape[0] &&
        def.n == instrShape[1] && def.k == instrShape[2])
      return def;
  }
  return std::nullopt;
}

inline int32_t encodeWmmaShape(SQMMALayout layoutA, SQMMALayout layoutB) {
  return (layoutA == SQMMALayout::col ? 2 : 0) |
         (layoutB == SQMMALayout::col ? 1 : 0);
}

inline int32_t getWmmaFmt(Type) { return 1; }
inline int32_t getWmmaFmt(SQMMAEltType) { return 1; }

inline bool needsWmmaScaleOperands(SQMMAEltType eltType) {
  switch (eltType) {
  case SQMMAEltType::e4m3:
  case SQMMAEltType::e5m2:
    return true;
  default:
    return false;
  }
}

inline SQMMALayout flipWmmaLayout(SQMMALayout layout) {
  return layout == SQMMALayout::row ? SQMMALayout::col : SQMMALayout::row;
}

inline SQMMALayout getDefaultWmmaFragmentLayout(unsigned opIdx) {
  assert(opIdx < 2 && "WMMA operand index must be 0 or 1");
  return opIdx == 0 ? SQMMALayout::row : SQMMALayout::col;
}

inline SQMMALayout inferWmmaFragmentLayout(Value value, unsigned opIdx) {
  while (auto cvt = value.getDefiningOp<gpu::ConvertLayoutOp>())
    value = cvt.getSrc();
  while (auto bitcast = value.getDefiningOp<triton::BitcastOp>())
    value = bitcast.getSrc();
  if (auto trans = value.getDefiningOp<triton::TransOp>())
    return flipWmmaLayout(inferWmmaFragmentLayout(trans.getSrc(), opIdx));
  return getDefaultWmmaFragmentLayout(opIdx);
}

struct WmmaDotOperandContract {
  gpu::DotOperandEncodingAttr dotEncoding;
  LinearLayout linearLayout;
  unsigned opIdx;
  unsigned rank;
};

inline FailureOr<WmmaDotOperandContract>
resolveWmmaDotOperandContract(RankedTensorType tensorTy,
                              unsigned expectedOpIdx) {
  auto dotEncoding =
      dyn_cast<gpu::DotOperandEncodingAttr>(tensorTy.getEncoding());
  if (!dotEncoding)
    return failure();
  if (dotEncoding.getOpIdx() != expectedOpIdx)
    return failure();
  return WmmaDotOperandContract{
      dotEncoding, dotEncoding.toLinearLayout(tensorTy.getShape()),
      expectedOpIdx, static_cast<unsigned>(tensorTy.getRank())};
}

inline Value extractWmmaOperandVectorFromContract(
    Location loc, Value operand, const WmmaDotOperandContract &contract,
    const LLVMTypeConverter *typeConverter, RewriterBase &rewriter, int batch,
    int nonK, int kIdx, int kInst, int kBase, int kPadding, Type elemTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = unpackLLElements(loc, operand, rewriter);

  auto outDimNames = contract.linearLayout.getOutDimNames();
  auto *ctx = (*outDimNames.begin()).getContext();
  StringAttr dim0 = StringAttr::get(ctx, "dim0");
  StringAttr dim1 = StringAttr::get(ctx, "dim1");
  StringAttr dim2 = StringAttr::get(ctx, "dim2");

  const int kElemIdx = kIdx * kInst;
  const int mCoord = (contract.opIdx == 0) ? nonK : kElemIdx;
  const int nCoord = (contract.opIdx == 0) ? kElemIdx : nonK;

  SmallVector<std::pair<StringAttr, int32_t>> outCoords;
  if (contract.rank == 3)
    outCoords = {{dim0, batch}, {dim1, mCoord}, {dim2, nCoord}};
  else
    outCoords = {{dim0, mCoord}, {dim1, nCoord}};

  auto inDims = contract.linearLayout.pseudoinvert().apply(outCoords);
  const int startReg = inDims[0].second;

  Type llvmElemTy = typeConverter->convertType(elemTy);
  Type vecTy = vec_ty(llvmElemTy, kBase);
  Value vec = b.undef(vecTy);
  const int validK = kBase - kPadding;
  Value zero = LLVM::ZeroOp::create(rewriter, loc, llvmElemTy);
  for (int k = 0; k < kBase; ++k) {
    Value v = (k < validK) ? elems[startReg + k] : zero;
    vec = b.insert_element(vecTy, vec, v, b.i32_val(k));
  }
  return vec;
}

inline SmallVector<Value>
buildWmmaIntrinsicArgs(Location loc, Value opA, Value opB, Value opC,
                       SQMMALayout layoutA, SQMMALayout layoutB,
                       const WmmaIntrinsicSignature &signature,
                       RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> args = {opA,
                             opB,
                             opC,
                             b.i32_val(signature.sat),
                             b.i32_val(getWmmaFmt(signature.eltType)),
                             b.i32_val(encodeWmmaShape(layoutA, layoutB))};
  if (needsWmmaScaleOperands(signature.eltType)) {
    args.push_back(b.i32_val(0));
    args.push_back(b.i32_val(0));
    args.push_back(b.i32_val(0));
  }
  return args;
}

inline SmallVector<Value>
buildWmmaIntrinsicArgs(Location loc, Value opA, Value opB, Value opC,
                       SQMMALayout layoutA, SQMMALayout layoutB,
                       SQMMAEltType eltType, RewriterBase &rewriter) {
  return buildWmmaIntrinsicArgs(
      loc, opA, opB, opC, layoutA, layoutB,
      WmmaIntrinsicSignature{eltType, /*m=*/0, /*n=*/0, /*k=*/0,
                             /*name=*/nullptr, /*sat=*/1},
      rewriter);
}

inline SmallVector<Value>
buildWmmaIntrinsicArgs(Location loc, Value opA, Value opB, Value opC,
                       SQMMALayout layoutA, SQMMALayout layoutB, Type elemTy,
                       RewriterBase &rewriter) {
  auto eltType = getWmmaEltType(elemTy);
  assert(eltType && "WMMA element type must be validated before arg building");
  return buildWmmaIntrinsicArgs(loc, opA, opB, opC, layoutA, layoutB, *eltType,
                                rewriter);
}

inline std::string getSqmmaTypeTag(SQMMAEltType type) {
  switch (type) {
  case SQMMAEltType::f16:
    return "fmma";
  case SQMMAEltType::bf16:
    return "bfmma";
  case SQMMAEltType::tf32:
    return "tfmma";
  case SQMMAEltType::s8:
    return "smma";
  case SQMMAEltType::e4m3:
    return "e4m3";
  case SQMMAEltType::e5m2:
    return "e5m2";
  default:
    return "";
  }
}

inline bool isSupportedSqmmaInstrMN(unsigned m, unsigned n) {
  static constexpr std::pair<unsigned, unsigned> kAllowedMN[] = {
      {32, 32}, {32, 64},  {32, 128}, {16, 64},  {64, 16},   {64, 32},
      {64, 64}, {64, 128}, {128, 32}, {128, 64}, {128, 128},
  };
  for (const auto &[supportedM, supportedN] : kAllowedMN) {
    if (supportedM == m && supportedN == n)
      return true;
  }
  return false;
}

inline bool isSupportedSqmmaInstrMN(SQMMAEltType eltType, unsigned m,
                                    unsigned n) {
  switch (eltType) {
  case SQMMAEltType::f16:
  case SQMMAEltType::bf16:
  case SQMMAEltType::s8:
  case SQMMAEltType::e4m3:
  case SQMMAEltType::e5m2:
    return isSupportedSqmmaInstrMN(m, n);
  case SQMMAEltType::tf32: {
    static constexpr std::pair<unsigned, unsigned> kAllowedTf32MN[] = {
        {16, 64}, {32, 32}, {32, 64},  {64, 16},
        {64, 32}, {64, 64}, {128, 64}, {128, 128},
    };
    for (const auto &[supportedM, supportedN] : kAllowedTf32MN) {
      if (supportedM == m && supportedN == n)
        return true;
    }
    return false;
  }
  default:
    return false;
  }
}

inline bool isSupportedSqmma(SQMMAEltType eltTypeA, SQMMAEltType eltTypeB,
                             SQMMAEltType eltTypeC, unsigned m, unsigned n,
                             unsigned k) {
  if (m == 0 || n == 0 || k == 0 || (m % 8) || (n % 8) || (k % 8))
    return false;
  if (eltTypeA != eltTypeB)
    return false;
  if (!isSupportedSqmmaInstrMN(eltTypeA, m, n))
    return false;

  auto isValidPh1K = [&](SQMMAEltType type, unsigned mVal, unsigned nVal,
                         unsigned kVal) {
    switch (type) {
    case SQMMAEltType::f16:
    case SQMMAEltType::bf16:
      if (kVal == 16 || kVal == 32 || kVal == 64)
        return true;
      return kVal == 128 &&
             ((mVal == 16 && nVal == 64) || (mVal == 64 && nVal == 16));
    case SQMMAEltType::tf32:
      return kVal == 8 || kVal == 16 || kVal == 32;
    case SQMMAEltType::s8:
    case SQMMAEltType::e4m3:
    case SQMMAEltType::e5m2:
      return kVal == 32 || kVal == 64 || kVal == 128;
    default:
      return false;
    }
  };

  switch (eltTypeA) {
  case SQMMAEltType::f16:
  case SQMMAEltType::bf16:
  case SQMMAEltType::tf32:
    return eltTypeC == SQMMAEltType::f32 && isValidPh1K(eltTypeA, m, n, k);
  case SQMMAEltType::s8:
    return eltTypeC == SQMMAEltType::s32 && isValidPh1K(eltTypeA, m, n, k);
  case SQMMAEltType::e4m3:
  case SQMMAEltType::e5m2:
    return eltTypeC == SQMMAEltType::f32 && isValidPh1K(eltTypeA, m, n, k);
  default:
    return false;
  }
}

inline std::string lookupSqmmaIntrinsic(SQMMAEltType type, unsigned m,
                                        unsigned n, unsigned k) {
  auto tag = getSqmmaTypeTag(type);
  if (tag.empty())
    return "";
  return ("llvm.musa.sqmma." + tag + ".m" + std::to_string(m) + "n" +
          std::to_string(n) + "k" + std::to_string(k) + ".mma");
}

inline Value materializeUseCFlag(Location loc, Value useC,
                                 RewriterBase &rewriter) {
  if (useC)
    return useC;
  return arith::ConstantIntOp::create(rewriter, loc, 1, 1);
}

inline Value selectAccumulatorValue(Location loc, Value useC, Value acc,
                                    Value zero, RewriterBase &rewriter) {
  auto constUseC = ::mlir::triton::getBoolFromConstant(useC);
  if (constUseC && *constUseC)
    return acc;
  if (constUseC && !*constUseC)
    return zero;
  return LLVM::SelectOp::create(rewriter, loc, acc.getType(), useC, acc, zero);
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_MMA_CONTRACT_UTILS_H
