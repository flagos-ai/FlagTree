#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct MakeTensorPtrOpConversion
    : public ConvertOpToLLVMPattern<triton::MakeTensorPtrOp> {
  using ConvertOpToLLVMPattern<triton::MakeTensorPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> elems;
    elems.append(adaptor.getOffsets().begin(), adaptor.getOffsets().end());
    elems.append(adaptor.getShape().begin(), adaptor.getShape().end());
    elems.append(adaptor.getStrides().begin(), adaptor.getStrides().end());
    elems.push_back(adaptor.getBase());

    Value packed = ::mlir::packLLElements(op.getLoc(), getTypeConverter(),
                                          elems, rewriter, op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

struct AdvanceOpConversion : public ConvertOpToLLVMPattern<triton::AdvanceOp> {
  using ConvertOpToLLVMPattern<triton::AdvanceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    auto elems = ::mlir::unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto offsets = adaptor.getOffsets();

    for (auto [i, offset] : llvm::enumerate(offsets)) {
      elems[i] = b.add(offset, elems[i]);
    }

    Value packed = ::mlir::packLLElements(loc, getTypeConverter(), elems,
                                          rewriter, op.getPtr().getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion, AdvanceOpConversion>(typeConverter,
                                                               benefit);
}
