#include "PatternPopulation.h"

#include "PTXAsmFormat.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

// Zero-pad chunks with fewer than four packed bytes, then retain
// only the unpacked values that correspond to real input bytes.
// Convert eight packed E2M1 values into four registers containing eight BF16
// values. IFWA accepts this PTX-compatible bit-manipulation sequence.
static constexpr const char *FP4ToBF16Asm =
    "{\n"
    ".reg .b32 a<14>;\n"
    "and.b32  a0, $4, -2004318072;\n\t"
    "shr.u32  a1, a0, 3;\n\t"
    "and.b32  a2, $4, 2004318071;\n\t"
    "shr.u32  a3, a2, 16;\n\t"
    "shr.u32  a4, a0, 19;\n\t"
    "prmt.b32 a5, -1065353216, -1065336832, a2;\n\t"
    "prmt.b32 a6, -1065353216, -1065336832, a3;\n\t"
    "prmt.b32 a7, 1061109504, 1077952576, a2;\n\t"
    "prmt.b32 a8, 1061109504, 1077952576, a3;\n\t"
    "prmt.b32 a9, 32768, 0, a1;\n\t"
    "prmt.b32 a10, 32768, 0, a4;\n\t"
    "or.b32   a11, a7, a9;\n\t"
    "or.b32   a12, a8, a10;\n\t"
    "prmt.b32 $0, a5, a11, 20800;\n\t"
    "prmt.b32 $1, a5, a11, 29538;\n\t"
    "prmt.b32 $2, a6, a12, 20800;\n\t"
    "prmt.b32 $3, a6, a12, 29538;\n\t"
    "}";

// Convert eight packed E2M1 values into four registers containing eight FP16
// values. This is used when dot_scaled selects FP16 as its compute type.
static constexpr const char *FP4ToFP16Asm =
    "{\n"
    ".reg .b32 a<11>;\n"
    ".reg .b16 t<4>;\n"
    "and.b32  a0, $4, 0x77777777;\n\t"
    "and.b32  a1, $4, 0x88888888;\n\t"
    "shr.u32  a2, a1, 3;\n\t"
    "shr.u32  a3, a0, 16;\n\t"
    "shr.u32  a4, a2, 16;\n\t"
    "prmt.b32 a5, 0x3C383000, 0x4C484440, a0;\n"
    "prmt.b32 a6, 0x3C383000, 0x4C484440, a3;\n"
    "prmt.b32 a7, 0x00008000, 0x0, a2;\n"
    "prmt.b32 a8, 0x00008000, 0x0, a4;\n"
    "or.b32   a9, a5, a7;\n\t"
    "or.b32   a10, a6, a8;\n\t"
    "mov.b32  {t0, t1}, a9;\n"
    "mov.b32  {t2, t3}, a10;\n"
    "cvt.rn.f16x2.e4m3x2 $0, t0;\n"
    "cvt.rn.f16x2.e4m3x2 $1, t1;\n"
    "cvt.rn.f16x2.e4m3x2 $2, t2;\n"
    "cvt.rn.f16x2.e4m3x2 $3, t3;\n"
    "}";

Value createInlineAsmUpcast(Location loc, RewriterBase &rewriter, bool toFp16,
                            Type retType, Value packedVec) {
  GPNPUPTXBuilder builder;
  SmallVector<GPNPUPTXBuilder::Operand *> operands;
  for (int i = 0; i < 4; ++i)
    operands.push_back(builder.newOperand("=r"));
  operands.push_back(builder.newOperand(packedVec, "r"));
  auto &asmOp = *builder.create(toFp16 ? FP4ToFP16Asm : FP4ToBF16Asm);
  asmOp(operands, /*onlyAttachMLIRArgs=*/true);
  return builder.launch(rewriter, loc, retType, false);
}

class GPNPUFp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  using ConvertOpToLLVMPattern<Fp4ToFpOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Required by the struct_ty macro below.
    MLIRContext *ctx = op.getContext();
    Type elemType = op.getType().getElementType();
    if (elemType != f16_ty && elemType != bf16_ty)
      return rewriter.notifyMatchFailure(op,
                                         "GPNPU FP4 upcast requires FP16/BF16");

    SmallVector<Value> packedValues =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> results;
    results.reserve(packedValues.size() * 2);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (size_t base = 0; base < packedValues.size(); base += 4) {
      size_t chunkSize = std::min<size_t>(4, packedValues.size() - base);
      Value packedVec = b.undef(vec_ty(i8_ty, 4));
      for (size_t i = 0; i < 4; ++i) {
        Value value = i < chunkSize ? packedValues[base + i] : b.i8_val(0);
        packedVec =
            b.insert_element(packedVec, value, b.i32_val(static_cast<int>(i)));
      }

      Type retType = struct_ty(SmallVector<Type>(4, i32_ty));
      Value unpacked = createInlineAsmUpcast(loc, rewriter, elemType == f16_ty,
                                             retType, packedVec);
      for (size_t i = 0; i < 4; ++i) {
        Value pair = b.bitcast(b.extract_val(unpacked, i), vec_ty(elemType, 2));
        for (size_t lane = 0; lane < 2; ++lane) {
          if (2 * i + lane < 2 * chunkSize)
            results.push_back(
                b.extract_element(pair, b.i32_val(static_cast<int>(lane))));
        }
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::Intellifusion::populateGPNPUFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GPNPUFp4ToFpOpPattern>(typeConverter, benefit);
}
