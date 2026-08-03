#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/PutMemOrValueOpToLLVM.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace mlir;
namespace tle = mlir::triton::tle;
namespace triton = mlir::triton;
namespace ttg = mlir::triton::gpu;

Value getDistDevicePtr(tle::PutMemOrValueOp op, SmallVector<Value> &srcElems) {
  if (!srcElems.empty())
    return srcElems[0];
  else {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    return func.getArgument(1);
  }
}

struct PutMemOrValueOpConversion
    : public ConvertOpToLLVMPattern<tle::PutMemOrValueOp> {
  PutMemOrValueOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::PutMemOrValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value> srcElems;
    auto loc = op.getLoc();
    if (auto comm = op.getComm())
      srcElems = unpackLLElements(loc, comm, rewriter);
    auto comm = getDistDevicePtr(op, srcElems);
    auto peer = unpackLLElements(loc, op.getPeer(), rewriter)[0];
    auto teamKind = tle::getTeamKindValue(op.getTeamKindAttr());
    auto coopKind = tle::getTeamKindValue(op.getCoopKindAttr());
    auto putType = op.getPutTypeAttr();
    auto value = op.getValue();
    // auto dst = op.getDst();
    // auto dstOffset = op.getDstOffset();
    auto pid = rewriter.create<triton::GetProgramIdOp>(
        loc, rewriter.getI32Type(), triton::ProgramIDDim::X);

    llvm::errs() << "[PutMemOrValueOpConversion]" << pid << "\n";
    // tle::getPutsFuncCall(loc, rewriter, comm,
    //                          teamKind, peer, Value dst,
    //                          size_t dstOffset, value, coopKind,
    //                          putType);
    return success();
  }
};

} // namespace

namespace mlir::triton::tle {

void populatePutMemOrValueOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit) {
  patterns.add<PutMemOrValueOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::tle
