#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    Value accum = c;
    Type tgtTy = accum.getType();
    auto castToTargetType = [&](Value v) -> Value {
      if (v.getType() == tgtTy)
        return v;
      if (isa<FloatType>(tgtTy) && isa<FloatType>(v.getType())) {
        auto srcTy = cast<FloatType>(v.getType());
        auto dstTy = cast<FloatType>(tgtTy);
        if (srcTy.getWidth() < dstTy.getWidth())
          return LLVM::FPExtOp::create(builder, loc, tgtTy, v);
        return LLVM::FPTruncOp::create(builder, loc, tgtTy, v);
      }
      if (isa<IntegerType>(tgtTy) && isa<IntegerType>(v.getType())) {
        auto srcTy = cast<IntegerType>(v.getType());
        auto dstTy = cast<IntegerType>(tgtTy);
        if (srcTy.getWidth() < dstTy.getWidth())
          return LLVM::SExtOp::create(builder, loc, tgtTy, v);
        return LLVM::TruncOp::create(builder, loc, tgtTy, v);
      }
      llvm_unreachable("unsupported type conversion in FMA dot lowering");
    };
    for (auto it = llvm::zip(a, b).begin(); it != llvm::zip(a, b).end(); ++it) {
      Value aElem = castToTargetType(std::get<0>(*it));
      Value bElem = castToTargetType(std::get<1>(*it));

      // to avoid: 'llvm.intr.fmuladd' op operand #0 must be floating point LLVM
      // type or LLVM dialect-compatible vector of floating point LLVM type, but
      // got 'i32'
      llvm::TypeSwitch<Type>(tgtTy)
          .Case<FloatType>([&](auto) {
            accum = LLVM::FMulAddOp::create(builder, loc, aElem, bElem, accum);
          })
          .Case<IntegerType>([&](auto) {
            accum = LLVM::AddOp::create(
                builder, loc, LLVM::MulOp::create(builder, loc, aElem, bElem),
                accum);
          });
    }
    return accum;
  }
};

} // namespace

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
