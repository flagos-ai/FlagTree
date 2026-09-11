//===- RawOpToLLVM.cpp - lower triton_xpu.raw to an inlined call ----------===//
//
// `triton_xpu.raw` carries a user-provided device function as LLVM IR text.
// This pattern splices that payload into the enclosing module and replaces the
// op with an `always_inline` `llvm.call`, so the payload ends up in the same
// kernel as the surrounding Triton code once the LLVM inliner runs.
//
// This is the xpu3 cluster counterpart of `sdnn.raw`'s lowering in
// TritonSDNNToLLVM.cpp, and mirrors jupiter's RawOpToLLVM.cpp for the jupiter
// cluster path. The payload parsing and symbol splicing are the same; the
// calling convention is not -- there are no memrefs here, so every operand is
// passed through after type conversion.
//
//===----------------------------------------------------------------------===//

#include "PatternTritonXPUOpToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

namespace {

struct RawOpConversion : public ConvertOpToLLVMPattern<triton::xpu::RawOp> {
  RawOpConversion(const LLVMTypeConverter &typeConverter,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::RawOp>(typeConverter, benefit) {}

  // Accept either LLVM-dialect MLIR text or textual LLVM IR (what
  // `xpu-clang -S -emit-llvm` produces).
  static OwningOpRef<ModuleOp> parsePayload(MLIRContext *ctx,
                                            StringRef payload) {
    {
      // Swallow diagnostics from the MLIR attempt so a plain-.ll payload does
      // not produce spurious parse errors.
      ScopedDiagnosticHandler quiet(ctx,
                                    [](Diagnostic &) { return success(); });
      ParserConfig config(ctx);
      if (auto mod = parseSourceString<ModuleOp>(payload, config))
        return mod;
    }

    llvm::LLVMContext llvmCtx;
    llvm::SMDiagnostic err;
    auto llvmMod = llvm::parseIR(
        llvm::MemoryBufferRef(payload, "triton_xpu.raw"), err, llvmCtx);
    if (!llvmMod)
      return nullptr;
    return translateLLVMIRToModule(std::move(llvmMod), ctx);
  }

  // Splice the symbols carried by `llvm_ir` into the enclosing module.
  // Idempotent across several raw ops sharing the same payload.
  static LogicalResult materializePayload(ModuleOp mod, StringRef payload,
                                          StringRef callee, Location loc) {
    if (mod.lookupSymbol<LLVM::LLVMFuncOp>(callee))
      return success();

    OwningOpRef<ModuleOp> payloadMod = parsePayload(mod.getContext(), payload);
    if (!payloadMod)
      return emitError(loc)
             << "triton_xpu.raw: cannot parse llvm_ir payload as "
                "either MLIR or LLVM IR";

    OpBuilder builder(mod.getContext());
    builder.setInsertionPointToStart(mod.getBody());
    for (Operation &op : payloadMod->getOps()) {
      auto sym = dyn_cast<SymbolOpInterface>(&op);
      if (!sym || mod.lookupSymbol(sym.getNameAttr()))
        continue;
      Operation *cloned = builder.clone(op);
      // Payload functions are device helpers, not kernel entry points. Say so
      // in the linkage, not just in an attribute: the backend picks the kernel
      // out of the module as "the external definition", so an external payload
      // definition would be mistaken for one. Internal also lets LLVM drop the
      // body once always_inline has fired.
      if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(cloned)) {
        funcOp->setAttr(triton::xpu::kRawPayloadAttrName,
                        builder.getUnitAttr());
        if (!funcOp.isExternal()) {
          funcOp.setLinkage(LLVM::Linkage::Internal);
          // xpu-clang marks its device functions `hidden`, and LLVM asserts
          // that local linkage implies default visibility, so the visibility
          // has to go with the linkage change.
          funcOp.setVisibility_(LLVM::Visibility::Default);
        }
      }
    }

    if (!mod.lookupSymbol<LLVM::LLVMFuncOp>(callee))
      return emitError(loc) << "triton_xpu.raw: callee '" << callee
                            << "' not defined by the llvm_ir payload";
    return success();
  }

  LogicalResult
  matchAndRewrite(triton::xpu::RawOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    StringRef callee = op.getCallee();

    if (op.getLlvmIr().empty() &&
        op->hasAttr(triton::xpu::kRawSourceIdAttrName))
      return op.emitError("triton_xpu.raw: deferred payload '")
             << callee
             << "' was never materialized; the backend must run "
                "tritonxpu-materialize-deferred-raw before lowering to LLVM";
    if (failed(materializePayload(mod, op.getLlvmIr(), callee, loc)))
      return failure();
    auto funcOp = mod.lookupSymbol<LLVM::LLVMFuncOp>(callee);

    // Every operand is passed through: `tt.ptr<T>` has already become
    // `!llvm.ptr` and scalars keep their type. A tensor operand would arrive
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

    if (funcOp.getNumArguments() != args.size())
      return op.emitError("triton_xpu.raw: callee '")
             << callee << "' takes " << funcOp.getNumArguments()
             << " arguments but " << args.size() << " were passed";

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
