#include "Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"

#include "IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/StringSwitch.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
namespace tle = mlir::triton::iluvatar_tle;

constexpr llvm::StringLiteral kSpaceAttr = "space";
constexpr llvm::StringLiteral kOrderAttr = "order";
constexpr llvm::StringLiteral kIndexAttr = "barrier_index";
constexpr llvm::StringLiteral kGroupKindAttr = "group_kind";

Value getDistDevicePtr(tle::DistributedBarrierOp op,
                       SmallVector<Value> &srcElems) {
  if (!srcElems.empty())
    return srcElems[0];
  else {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    // arg0: memory pointer
    // arg1: communicator pointer
    return func.getArgument(1);
  }
}

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::DistributedBarrierOp> {
  using ConvertOpToLLVMPattern<
      tle::DistributedBarrierOp>::ConvertOpToLLVMPattern;

  LogicalResult lowerClusterBarrier(tle::DistributedBarrierOp op,
                                    ConversionPatternRewriter &rewriter) const {
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  lowerDeviceSpaceBarrier(tle::DistributedBarrierOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto kindAttr = op->getAttrOfType<StringAttr>(kGroupKindAttr);
    auto orderAttr = op.getOrderAttr();
    auto indexAttr = op->getAttrOfType<IntegerAttr>(kIndexAttr);
    auto loc = op.getLoc();
    SmallVector<Value> srcElems;
    // flagcxCoopKind_t is thread=0, warp=1, block=2, tile_span=3, lanes=4 (see
    // flagcx/adaptor/include/device_api/flagcx_device_enums.h). Only the three
    // the corex adapter can build from the scalar barrier ABI are accepted:
    // tile_span and lanes need span/mask operands the ABI does not carry, and
    // there is no grid coop kind at all, so a "grid" device barrier would land
    // on tile_span and silently synchronize less than it was asked to.
    auto getCoopKind =
        [](StringRef kind) -> std::optional<tle::FlagCXCoopKind> {
      return llvm::StringSwitch<std::optional<tle::FlagCXCoopKind>>(kind)
          .Case("thread", tle::FlagCXCoopKind::THREAD)
          .Case("warp", tle::FlagCXCoopKind::WARP)
          .Case("block", tle::FlagCXCoopKind::BLOCK)
          .Default(std::nullopt);
    };
    auto coopKind = getCoopKind(kindAttr.getValue());
    if (!coopKind)
      return rewriter.notifyMatchFailure(op, "invalid coop_kind");

    if (auto src = adaptor.getSrc())
      srcElems = unpackLLElements(loc, src, rewriter);

    auto comm = getDistDevicePtr(op, srcElems);
    auto coopKindAttr =
        tle::FlagCXCoopKindAttr::get(rewriter.getContext(), *coopKind);
    auto barrierTypeAttr = op.getBarrierTypeAttr();
    auto multimemAttr = rewriter.getBoolAttr(false);
#ifdef FLAGCX_ENABLED
    rewriter.replaceOpWithNewOp<tle::DeviceIntraBarrierOp>(
        op, comm, barrierTypeAttr, coopKindAttr, indexAttr, multimemAttr,
        orderAttr);
#endif
    return success();
  }

  LogicalResult
  matchAndRewrite(tle::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto spaceAttr = op->getAttrOfType<StringAttr>(kSpaceAttr))
      if (spaceAttr.getValue() == "device")
        return lowerDeviceSpaceBarrier(op, adaptor, rewriter);

    if (auto kindAttr = op->getAttrOfType<StringAttr>(kGroupKindAttr)) {
      // Both need synchronization across CTAs: grid relies on a cooperative
      // launch and submesh on CTA clusters. Corex provides neither, and
      // degrading them to a CTA barrier would silently drop the cross-CTA
      // guarantee, so reject them instead.
      if (kindAttr.getValue() == "grid")
        return op.emitOpError(
            "grid distributed barrier is not supported on corex: it requires "
            "cooperative grid launch");
      if (kindAttr.getValue() == "submesh")
        return op.emitOpError(
            "sub-mesh distributed barrier is not supported on corex: it "
            "requires CTA cluster launch");
      return lowerClusterBarrier(op, rewriter);
    }
    return lowerClusterBarrier(op, rewriter);
  }
};

} // namespace

void mlir::triton::iluvatar_tle::populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, benefit);
}
