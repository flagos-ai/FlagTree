#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

#include <map>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::IluvatarMmaEncodingAttr;

namespace {

using ValueTable = std::map<std::pair<int, int>, Value>;

ValueTable extractLoadedOperand(Value llStruct, int repOuter, int repK,
                                Type elemTy, int elemsPerTCUPack, Location loc,
                                ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  ValueTable rcds;
  SmallVector<Value> elems = unpackLLElements(loc, llStruct, rewriter);

  assert(static_cast<int>(elems.size()) == repOuter * repK * elemsPerTCUPack &&
         "unexpected number of scalar TCU operand values");

  // Generic LinearLayout conversion provides scalar elements; pack them into
  // x4 operands consumed by the TCU intrinsic.
  Type packTy = vec_ty(elemTy, elemsPerTCUPack);
  int offset = 0;
  for (int outer = 0; outer < repOuter; ++outer) {
    for (int k = 0; k < repK; ++k) {
      Value pack = b.undef(packTy);
      for (int i = 0; i < elemsPerTCUPack; ++i)
        pack = b.insert_element(packTy, pack, elems[offset++], b.i32_val(i));
      rcds[{outer, k}] = pack;
    }
  }
  return rcds;
}

std::pair<Value, RankedTensorType>
getTCUOperand(Value operand, Value convertedOperand,
              ConversionPatternRewriter &rewriter) {
  auto operandTy = cast<RankedTensorType>(operand.getType());
  if (operandTy.getElementType().isF16())
    return {convertedOperand, operandTy};

  auto extOp = operand.getDefiningOp<arith::ExtFOp>();
  if (!extOp)
    return {convertedOperand, operandTy};

  auto sourceTy = dyn_cast<RankedTensorType>(extOp.getIn().getType());
  if (!sourceTy || !sourceTy.getElementType().isF16())
    return {convertedOperand, operandTy};

  Value convertedSource = rewriter.getRemappedValue(extOp.getIn());
  assert(convertedSource && "expected converted f16 TCU operand");
  return {convertedSource, sourceTy};
}

} // namespace

namespace mlir::triton::ILUVATAR {

LogicalResult convertTCU161616(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Value A = op.getA();
  Value B = op.getB();
  Value D = op.getResult();
  auto [convertedA, ATensorTy] = getTCUOperand(A, adaptor.getA(), rewriter);
  auto [convertedB, BTensorTy] = getTCUOperand(B, adaptor.getB(), rewriter);
  auto DTensorTy = cast<RankedTensorType>(D.getType());
  auto mmaLayout = cast<IluvatarMmaEncodingAttr>(DTensorTy.getEncoding());
  auto ALayout = cast<DotOperandEncodingAttr>(ATensorTy.getEncoding());
  auto BLayout = cast<DotOperandEncodingAttr>(BTensorTy.getEncoding());
  Type elemTy = ATensorTy.getElementType();

  assert(mmaLayout.isVolta() && "only Iluvatar TCU v1 is supported");
  assert(ATensorTy.getElementType() == BTensorTy.getElementType() &&
         ((DTensorTy.getElementType().isF32() &&
           (elemTy.isF16() || elemTy.isBF16() || elemTy.isF32())) ||
          (DTensorTy.getElementType().isInteger(32) && elemTy.isInteger(8))) &&
         "TCU currently supports f16/bf16/f32 inputs with f32 accum and i8 "
         "inputs with i32 accum");
  assert(ALayout.getOpIdx() == 0 && BLayout.getOpIdx() == 1 &&
         "unexpected Iluvatar TCU dot operand indices");

  auto aRep = mmaLayout.getRepForOperand(
      ATensorTy.getShape(), ATensorTy.getElementType().getIntOrFloatBitWidth(),
      ALayout.getKWidth(), ALayout.getOpIdx());
  auto bRep = mmaLayout.getRepForOperand(
      BTensorTy.getShape(), BTensorTy.getElementType().getIntOrFloatBitWidth(),
      BLayout.getKWidth(), BLayout.getOpIdx());
  assert(aRep.size() == 3 && bRep.size() == 3 &&
         "Iluvatar TCU operands use batch, outer, k reps");
  assert(aRep[0] == 1 && bRep[0] == 1 &&
         "batched Iluvatar TCU lowering is not supported yet");

  int rep_m = aRep[1];
  int rep_k = aRep[2];
  int rep_n = bRep[2];
  assert(rep_k == bRep[1] && "A/B K repetitions must match");

  int elemsPerTCUPack = elemTy.isInteger(8) ? 8 : 4;
  ValueTable has =
      extractLoadedOperand(convertedA, rep_m, rep_k, ATensorTy.getElementType(),
                           elemsPerTCUPack, loc, rewriter);
  ValueTable hbs =
      extractLoadedOperand(convertedB, rep_n, rep_k, BTensorTy.getElementType(),
                           elemsPerTCUPack, loc, rewriter);

  // Initialize accumulators with external values. In Triton 3.6, the
  // accumulator struct order is defined by LinearLayout unpacking.
  SmallVector<Value> acc = unpackLLElements(loc, adaptor.getC(), rewriter);
  assert(static_cast<int>(acc.size()) == rep_m * rep_n * 4 &&
         "unexpected number of TCU accumulator values");

  Type accElemTy = elemTy.isInteger(8) ? Type(i32_ty) : Type(f32_ty);
  Type elemX4Ty = vec_ty(accElemTy, 4);
  StringRef intrinsic;
  if (elemTy.isInteger(8))
    intrinsic = "llvm.bi.matrix.mad.i32x4.i8x8";
  else if (elemTy.isF16())
    intrinsic = "llvm.bi.matrix.mad.f32x4.f16x4";
  else if (elemTy.isBF16())
    intrinsic = "llvm.bi.matrix.mad.f32x4.bf16x4";
  else if (elemTy.isF32())
    intrinsic = "llvm.bi.matrix.mad.f32x4.f32x4";
  else
    llvm_unreachable("unsupported Iluvatar TCU operand type");

  auto callMMA = [&](unsigned m, unsigned n, unsigned k) {
    Value ha = has.at({m, k});
    Value hb = hbs.at({n, k});

    Value accVec = b.undef(elemX4Ty);
    // 3.2 used m-major accumulator slots. The current LinearLayout packing
    // exposes accumulator values in n-major repeat order.
    int accIdx = (n * rep_m + m) * 4;
    for (int i = 0; i < 4; ++i)
      accVec =
          b.insert_element(elemX4Ty, accVec, acc[accIdx + i], b.i32_val(i));

    Value res =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, elemX4Ty,
                                        ValueRange{ha, hb, accVec})
            .getResult(0);
    for (int i = 0; i < 4; ++i)
      acc[accIdx + i] = b.extract_element(accElemTy, res, b.i32_val(i));
  };

  for (unsigned k = 0; k < rep_k; ++k)
    for (unsigned m = 0; m < rep_m; ++m)
      for (unsigned n = 0; n < rep_n; ++n)
        callMMA(m, n, k);

  // res holds the same layout as acc.
  Value res = packLLElements(loc, typeConverter, acc, rewriter, DTensorTy);
  rewriter.replaceOp(op, res);
  return success();
}

} // namespace mlir::triton::ILUVATAR
