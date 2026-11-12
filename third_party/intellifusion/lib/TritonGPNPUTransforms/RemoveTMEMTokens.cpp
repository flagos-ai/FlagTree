#include "Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace Intellifusion {

using namespace mlir::triton::nvidia_gpu;

namespace {

void eraseResult(Operation *op, unsigned resultIdx, Value replacement) {
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       op->getResultTypes(), op->getAttrs());
  state.types.erase(std::next(state.types.begin(), resultIdx));
  OpBuilder b(op);

  if (auto segmentSizes =
          op->getAttrOfType<DenseI32ArrayAttr>("resultSegmentSizes")) {
    // Update resultSegmentSizes attribute if it exists
    SmallVector<int32_t> newSegmentSizes(segmentSizes.asArrayRef());
    int pos = 0;
    for (auto &segmentSize : newSegmentSizes) {
      if (pos == resultIdx) {
        segmentSize = 0;
        break;
      }
      pos += segmentSize;
    }
    state.attributes.set("resultSegmentSizes",
                         b.getDenseI32ArrayAttr(newSegmentSizes));
  }
  Operation *newOp = b.create(state);
  SmallVector<Value> replacements = newOp->getResults();
  replacements.insert(std::next(replacements.begin(), resultIdx), replacement);
  op->replaceAllUsesWith(replacements);
  op->erase();
}

void removeTMEMToken(Operation *op, Value dummy) {
  if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
    mmaOp.getAccDepMutable().clear();
    if (mmaOp.getToken())
      eraseResult(mmaOp, 0, dummy);
  } else if (auto store = dyn_cast<TMEMStoreOp>(op)) {
    store.getDepMutable().clear();
    if (store.getToken())
      eraseResult(store, 0, dummy);
  } else if (auto alloc = dyn_cast<TMEMAllocOp>(op)) {
    if (alloc.getToken())
      eraseResult(alloc, 1, dummy);
  } else if (auto load = dyn_cast<TMEMLoadOp>(op)) {
    load.getDepMutable().clear();
    if (load.getToken())
      eraseResult(load, 1, dummy);
  }
}

} // anonymous namespace

class GPNPURemoveTMEMTokensPass
    : public PassWrapper<GPNPURemoveTMEMTokensPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPNPURemoveTMEMTokensPass)

  StringRef getArgument() const final { return "gpnpu-remove-tmem-tokens"; }
  StringRef getDescription() const final {
    return "remove GPNPU TMEM dependency tokens after scheduling";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, gpu::TritonGPUDialect,
                    nvidia_gpu::TritonNvidiaGPUDialect, ub::UBDialect>();
  }

  void runOnOperation() override {
    for (auto func : getOperation().getOps<FuncOp>()) {
      auto b = OpBuilder::atBlockBegin(&func.getBody().front());
      // Placeholder value that will get DCE'd by the canonicalizer.
      Value dummy = ub::PoisonOp::create(
          b, func.getLoc(), b.getType<triton::gpu::AsyncTokenType>());
      func.walk([&](Operation *op) { removeTMEMToken(op, dummy); });
    }
  }
};

std::unique_ptr<Pass> createGPNPURemoveTMEMTokensPass() {
  return std::make_unique<GPNPURemoveTMEMTokensPass>();
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
