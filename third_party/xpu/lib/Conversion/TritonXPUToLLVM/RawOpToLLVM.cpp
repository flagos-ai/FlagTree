//===- RawOpToLLVM.cpp - lower triton_xpu.raw to an inlined call ----------===//
//
// `triton_xpu.raw` names a user-provided device function. This pattern declares
// it and replaces the op with an `always_inline` `llvm.call`.
//
// The payload body is merged into the enclosing module later, on the LLVM 19
// side (`tle/raw/merge.py`), which is what makes the call inline: LLVM honours
// `alwaysinline` within one module, and a payload compiled into its own object
// file would keep a real ABI boundary at the call.
//
// This is the xpu3 cluster counterpart of `sdnn.raw`'s lowering in
// TritonSDNNToLLVM.cpp, and mirrors the alternate cluster path's
// RawOpToLLVM.cpp for that path. The declaration is built the same way; the
// calling convention is not -- there are no memrefs here, so every operand is
// passed through after type conversion.
//
//===----------------------------------------------------------------------===//

#include "PatternTritonXPUOpToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Target/LLVMIR/IntrinsicAttrTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

struct RawOpConversion : public ConvertOpToLLVMPattern<triton::xpu::RawOp> {
  RawOpConversion(const LLVMTypeConverter &typeConverter,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::RawOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::xpu::RawOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    StringRef callee = op.getCallee();

    // Every operand is passed through: `tt.ptr<T>` has already become
    // `!llvm.ptr<1>` and scalars keep their type. A tensor operand would arrive
    // here as an LLVM struct of registers, which no payload can consume, so
    // reject it rather than emit a call the verifier would then complain about.
    SmallVector<Value> args;
    for (auto [orig, converted] :
         llvm::zip_equal(op.getArgs(), adaptor.getArgs())) {
      if (isa<RankedTensorType>(orig.getType()))
        return op.emitError("triton_xpu.raw: tensor operands are not supported "
                            "on the cluster path; pass a pointer instead");
      args.push_back(converted);
    }

    // The payload is merged into this module at the LLVM 19 stage, so all that
    // is needed here is a declaration to call: its signature is the operand
    // types, which is the ABI the payload has to match (the merge checks the
    // payload's definition against exactly this declaration). Several raw ops
    // on the same payload share one declaration.
    auto funcOp = mod.lookupSymbol<LLVM::LLVMFuncOp>(callee);
    if (!funcOp) {
      SmallVector<Type> argTys;
      for (Value arg : args)
        argTys.push_back(arg.getType());
      OpBuilder builder(mod.getContext());
      builder.setInsertionPointToStart(mod.getBody());
      // The one gate (see IntrinsicAttrTable.h): the payload function is not a
      // table name, so it is created plain.
      funcOp = mlir::intrinsic_attr_table::getOrCreateDeclaration(
          builder, mod, loc, callee,
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(mod.getContext()),
                                      argTys));
    }

    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    callOp.setAlwaysInline(true);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateRawOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<RawOpConversion>(typeConverter, benefit);
}
