#include "TritonMETAXGPUTransforms/AsyncCopyUtils.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::METAX;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

namespace {

class LegalizeAsyncCopy
    : public TritonMETAXGPULegalizeAsyncCopyBase<LegalizeAsyncCopy> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ttg::AsyncCopyGlobalToLocalOp> copies;
    module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) { copies.push_back(copy); });
    if (copies.empty())
      return;

    ModuleAxisInfoAnalysis axisInfo(module);
    for (auto copy : copies) {
      auto srcTy = copy.getSrc().getType();
      auto dstTy = copy.getResult().getType();
      auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
      auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
      if (!blocked || !shared) {
        copy.emitOpError(
            "requires blocked source and swizzled shared destination");
        return signalPassFailure();
      }
      auto *info = axisInfo.getAxisInfo(copy.getSrc());
      auto replacement = chooseAsyncCopyLayout(srcTy, dstTy, info);
      if (!replacement) {
        copy.emitOpError(
            "cannot legalize lane-to-shared addresses for MetaX async copy");
        return signalPassFailure();
      }
      if (replacement == blocked)
        continue;

      OpBuilder builder(copy);
      for (OpOperand &operand : copy->getOpOperands()) {
        auto type = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!type)
          continue;
        auto converted = builder.create<ttg::ConvertLayoutOp>(
            copy.getLoc(), type.cloneWithEncoding(replacement), operand.get());
        operand.set(converted);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonMETAXGPULegalizeAsyncCopyPass() {
  return std::make_unique<LegalizeAsyncCopy>();
}
