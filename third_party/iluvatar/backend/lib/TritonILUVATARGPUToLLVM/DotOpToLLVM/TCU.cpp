#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include <map>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::IluvatarMmaEncodingAttr;

namespace {

using ValueTable = std::map<std::pair<int, int>, Value>;

// Element coordinate reached by register index `reg` with lane/warp/block = 0.
// Deriving repeat positions from the layout is deliberate: the order in which
// repeat bits sit in the register basis differs between A, B and the
// accumulator, and the N repeats deliberately sit *below* the warp bases so
// that each warp owns a contiguous column slab (see foldNRepsIntoRegister).
// Assuming a fixed outer-major/k-major flattening, or a uniform per-repeat
// element stride, is a trap: the M repeats step by instrShape[0] *
// warpsPerCTA[0] while the N repeats step by instrShape[1].
std::pair<int, int> registerCoord(const triton::LinearLayout &ll,
                                  StringAttr kRegister, StringAttr dim0,
                                  StringAttr dim1, int reg) {
  int c0 = 0, c1 = 0;
  for (int bit = 0; (reg >> bit) != 0; ++bit) {
    if (!((reg >> bit) & 1))
      continue;
    c0 ^= ll.getBasis(kRegister, bit, dim0);
    c1 ^= ll.getBasis(kRegister, bit, dim1);
  }
  return {c0, c1};
}

// Register indices grouped by (outer, k) element coordinate. Because the
// coordinates form a Cartesian grid, walking the sorted map yields outer-major
// / k-minor repeat order whatever the register basis order happens to be.
std::map<std::pair<int, int>, int>
groupRegistersByRep(const triton::LinearLayout &ll, MLIRContext *ctx, int rank,
                    bool outerIsDim0, int numRegisters, int step) {
  auto kRegister = StringAttr::get(ctx, "register");
  auto dimNames = standardOutDimNames(ctx, rank);
  auto dim0 = dimNames[rank - 2];
  auto dim1 = dimNames[rank - 1];

  std::map<std::pair<int, int>, int> byCoord;
  for (int reg = 0; reg < numRegisters; reg += step) {
    auto [c0, c1] = registerCoord(ll, kRegister, dim0, dim1, reg);
    byCoord[outerIsDim0 ? std::make_pair(c0, c1) : std::make_pair(c1, c0)] =
        reg;
  }
  return byCoord;
}

ValueTable extractLoadedOperand(Value llStruct, RankedTensorType operandTy,
                                int opIdx, int repOuter, int repK,
                                int elemsPerTCUPack, Location loc,
                                ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  ValueTable rcds;
  SmallVector<Value> elems = unpackLLElements(loc, llStruct, rewriter);

  assert(static_cast<int>(elems.size()) == repOuter * repK * elemsPerTCUPack &&
         "unexpected number of scalar TCU operand values");

  auto ll = cast<DotOperandEncodingAttr>(operandTy.getEncoding())
                .toLinearLayout(operandTy.getShape());
  // A is (M, K) so its outer dim is dim0; B is (K, N) so its outer dim is dim1.
  auto byCoord = groupRegistersByRep(ll, operandTy.getContext(),
                                     operandTy.getRank(), /*outerIsDim0=*/
                                     opIdx == 0, elems.size(), elemsPerTCUPack);
  assert(static_cast<int>(byCoord.size()) == repOuter * repK &&
         "TCU operand repeats do not tile the operand shape");

  // Generic LinearLayout conversion provides scalar elements; pack them into
  // x4/x8 operands consumed by the TCU intrinsic. The pack bits are the low
  // register bits, so every `elemsPerTCUPack`-aligned index starts one pack.
  Type packTy = vec_ty(operandTy.getElementType(), elemsPerTCUPack);
  int rep = 0;
  for (auto &[coord, offset] : byCoord) {
    Value pack = b.undef(packTy);
    for (int i = 0; i < elemsPerTCUPack; ++i)
      pack = b.insert_element(packTy, pack, elems[offset + i], b.i32_val(i));
    rcds[{rep / repK, rep % repK}] = pack;
    ++rep;
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
  if (ATensorTy.getElementType() != BTensorTy.getElementType()) {
    // Mixed-dtype dots unify operands via extf before tt.dot. Only peel extf
    // when both sides resolve to the same TCU operand type.
    convertedA = adaptor.getA();
    convertedB = adaptor.getB();
    ATensorTy = cast<RankedTensorType>(A.getType());
    BTensorTy = cast<RankedTensorType>(B.getType());
  }
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
  // kRotate relabels the reduced K axis, which only cancels out if both
  // operands agree on the relabeling. Any pass that reconstructs one operand's
  // encoding without carrying the flag over would otherwise silently compute
  // garbage.
  assert(ALayout.getKRotate() == BLayout.getKRotate() &&
         "Iluvatar TCU dot operands disagree on kRotate");

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
      extractLoadedOperand(convertedA, ATensorTy, /*opIdx=*/0, rep_m, rep_k,
                           elemsPerTCUPack, loc, rewriter);
  ValueTable hbs =
      extractLoadedOperand(convertedB, BTensorTy, /*opIdx=*/1, rep_n, rep_k,
                           elemsPerTCUPack, loc, rewriter);

  // Initialize accumulators with external values. In Triton 3.6, the
  // accumulator struct order is defined by LinearLayout unpacking.
  SmallVector<Value> acc = unpackLLElements(loc, adaptor.getC(), rewriter);
  assert(static_cast<int>(acc.size()) == rep_m * rep_n * 4 &&
         "unexpected number of TCU accumulator values");

  // Flat accumulator slot of each (m, n) repeat, in m-major / n-minor order.
  auto accByCoord = groupRegistersByRep(
      mmaLayout.toLinearLayout(DTensorTy.getShape()), DTensorTy.getContext(),
      DTensorTy.getRank(), /*outerIsDim0=*/true, acc.size(), /*step=*/4);
  assert(static_cast<int>(accByCoord.size()) == rep_m * rep_n &&
         "TCU accumulator repeats do not tile the result shape");
  SmallVector<int> accSlot;
  for (auto &[coord, offset] : accByCoord)
    accSlot.push_back(offset);

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
    int accIdx = accSlot[m * rep_n + n];
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
