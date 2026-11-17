// MIT License
// Copyright (c) 2026 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/MmaLayout.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::triton::metax {
namespace ttg = mlir::triton::gpu;

unsigned getMmaRegisterGroupLimit(int64_t size, unsigned atomSize,
                                  unsigned warps) {
  uint64_t atoms = llvm::divideCeil<uint64_t>(size, atomSize);
  return llvm::PowerOf2Ceil(
      std::max<uint64_t>(llvm::divideCeil(atoms, uint64_t(warps)), 1));
}

ttg::MACAMmaEncodingAttr buildMmaEncoding(MLIRContext *ctx,
                                          int computeCapability,
                                          ArrayRef<unsigned> warpsPerCTA,
                                          ArrayRef<unsigned> elementsMNK,
                                          unsigned colMajor) {
  return ttg::MACAMmaEncodingAttr::get(
      ctx, 2, computeCapability % 10, warpsPerCTA, elementsMNK, colMajor,
      /*isATrans=*/false, /*isBTrans=*/false, {1, 1});
}

std::optional<StringRef> checkMacaDotPreconditions(triton::DotOp dot,
                                                   int computeCapability,
                                                   int numWarps) {
  if (computeCapability < 80 || computeCapability >= 90)
    return "requires MACA MMA version 2";
  if (numWarps <= 0 || !llvm::isPowerOf2_64(numWarps))
    return "requires a positive power-of-two warp count";
  auto aTy = dot.getA().getType();
  auto bTy = dot.getB().getType();
  auto dTy = dot.getType();
  for (auto type : {aTy, bTy, dTy}) {
    if (type.getRank() != 2)
      return "requires rank-2 tensors";
    if (llvm::any_of(type.getShape(), [](int64_t dim) {
          return dim <= 0 || dim > std::numeric_limits<int>::max() ||
                 !llvm::isPowerOf2_64(dim);
        }))
      return "requires static power-of-two dimensions representable as i32";
  }
  if (!mlir::supportMMA(dot, 2, computeCapability % 10))
    return "dot dtype/precision is not supported by the target MMA "
           "instructions";
  return std::nullopt;
}

std::optional<StringRef>
checkMacaDotLayout(triton::DotOp dot, Attribute encoding, Attribute operandA,
                   Attribute operandB, int computeCapability, int numWarps) {
  if (auto error = checkMacaDotPreconditions(dot, computeCapability, numWarps))
    return error;
  auto aTy = dot.getA().getType();
  auto bTy = dot.getB().getType();
  auto dTy = dot.getType();
  for (Attribute attr : {encoding, operandA, operandB})
    if (!attr || attr.getContext() != dot.getContext())
      return "requires layouts from the dot's context";
  auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(encoding);
  if (!mma)
    return "requires a MACA MMA result layout";
  ArrayRef<unsigned> warps = mma.getWarpsPerCTA();
  ArrayRef<unsigned> elements = mma.getElementsMNK();
  auto powersOfTwo = [](ArrayRef<unsigned> values) {
    return llvm::all_of(values,
                        [](unsigned v) { return llvm::isPowerOf2_64(v); });
  };
  if (warps.size() != 2 || elements.size() != 3 || !powersOfTwo(warps) ||
      !powersOfTwo(elements) ||
      uint64_t(warps[0]) * warps[1] != unsigned(numWarps))
    return "requires power-of-two MMA geometry matching num-warps";
  Type aElem = aTy.getElementType(), bElem = bTy.getElementType();
  if (mma.getColMajor() > 1 ||
      (mma.getColMajor() && (!(aElem.isF16() || aElem.isBF16()) ||
                             !(bElem.isF16() || bElem.isBF16()))))
    return "column-major MMA supports FP16/BF16 operands only";

  // Check target fields and A/B layouts against the expected encoding.
  auto expectedMma = buildMmaEncoding(dot.getContext(), computeCapability,
                                      warps, elements, mma.getColMajor());
  if (mma != expectedMma ||
      operandA !=
          ttg::DotOperandEncodingAttr::get(dot.getContext(), 0, mma, aElem) ||
      operandB !=
          ttg::DotOperandEncodingAttr::get(dot.getContext(), 1, mma, bElem))
    return "MMA target fields or A/B operand layouts do not match";

  int64_t m = dTy.getShape()[0], n = dTy.getShape()[1], k = aTy.getShape()[1];
  auto atom = mma.getThreadShape(false);
  // Keep the existing M/N bounds until larger register groups are validated.
  auto maxElements = [&](int64_t size, unsigned axis) {
    return getMmaRegisterGroupLimit(size, atom[axis], warps[axis]);
  };
  if (elements[0] > maxElements(m, 0) || elements[1] > maxElements(n, 1))
    return "MMA M/N register groups exceed the currently supported shape "
           "bounds";
  unsigned nativeK = 1;
  if (aElem.isF32() && dot.getInputPrecision() == InputPrecision::TF32)
    nativeK = 2;
  else if (aElem.isF16() || aElem.isBF16())
    nativeK = 4;
  else if (aElem.isSignlessInteger(8))
    nativeK = computeCapability >= 86 ? 8 : 4;
  else if (isa<Float8E4M3FNType, Float8E5M2Type>(aElem))
    nativeK = 8;
  if (elements[2] < nativeK || uint64_t(elements[2]) * atom[2] > uint64_t(k))
    return "MMA K register group must fit the instruction and dot K dimension";

  int repM = ttg::getNumRepM(m, warps, elements[0]);
  int repN = ttg::getNumRepN(n, warps, elements[1]);
  int repK = ttg::getNumRepK(k, elements[2]);
  if (int64_t(repM) * warps[0] * atom[0] * elements[0] < m ||
      int64_t(repN) * warps[1] * atom[1] * elements[1] < n ||
      int64_t(repK) * atom[2] * elements[2] != k)
    return "MMA geometry must cover M/N and exactly cover K";

  // All geometry guards precede conversion to the LinearLayout model.
  (void)ttg::toLinearLayout(dTy.cloneWithEncoding(mma));
  (void)ttg::toLinearLayout(aTy.cloneWithEncoding(operandA));
  (void)ttg::toLinearLayout(bTy.cloneWithEncoding(operandB));
  return std::nullopt;
}

} // namespace mlir::triton::metax
