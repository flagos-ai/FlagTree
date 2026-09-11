#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/GluonC500AsyncCopyPlan.h"
#include "Gluon/Targets/GluonC500Layout.h"
#include "Gluon/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "metax-gluon-verify-layout-contracts"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace c500 = ::mlir::triton::gpu::metax::gluon::c500;
namespace gluon_layout = ::mlir::triton::gpu::metax::gluon;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONVERIFYLAYOUTCONTRACTS
#include "Gluon/Passes.h.inc"

namespace {

static LogicalResult verifyLocalLoadContract(ttg::LocalLoadOp loadOp) {
  auto resultTy = dyn_cast<RankedTensorType>(loadOp.getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(loadOp.getSrc().getType());
  if (!resultTy || !memDescTy)
    return success();

  bool requiresDotContract = gluon_layout::isSimpleDotLocalLoad(loadOp);
  return gluon_layout::verifyRegisterToSharedContract(
      loadOp, resultTy, memDescTy,
      requiresDotContract ? gluon_layout::LayoutContractRole::DotOperand
                          : gluon_layout::LayoutContractRole::RegisterSubview);
}

static LogicalResult verifyLocalStoreContract(ttg::LocalStoreOp storeOp) {
  auto srcTy = dyn_cast<RankedTensorType>(storeOp.getSrc().getType());
  auto dstTy = dyn_cast<ttg::MemDescType>(storeOp.getDst().getType());
  if (!srcTy || !dstTy)
    return success();

  LDBG("[local-store-sink] source=" << srcTy.getEncoding()
                                    << ", shared=" << dstTy.getEncoding()
                                    << ", location=" << storeOp.getLoc());
  return gluon_layout::verifyRegisterToSharedContract(
      storeOp, srcTy, dstTy,
      gluon_layout::LayoutContractRole::RegisterToSharedSink);
}

static LogicalResult verifyLocalAllocInitializerContract(
    ttg::LocalAllocOp allocOp) {
  Value src = allocOp.getSrc();
  if (!src)
    return success();

  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(allocOp.getType());
  if (!srcTy || !memDescTy)
    return success();

  LDBG("[local-store-sink] local_alloc init source=" << srcTy.getEncoding()
                                                    << ", shared="
                                                    << memDescTy.getEncoding()
                                                    << ", location="
                                                    << allocOp.getLoc());
  return gluon_layout::verifyRegisterToSharedContract(
      allocOp, srcTy, memDescTy,
      gluon_layout::LayoutContractRole::RegisterToSharedSink);
}

static LogicalResult verifySameEncodingOp(Operation *op) {
  // ODS already verifies SameOperandsAndResultEncoding. Keep only the
  // residual Elementwise convention for ops that do not carry that trait.
  if (!op->hasTrait<OpTrait::Elementwise>() ||
      op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      isa<ttg::ConvertLayoutOp, tt::AddPtrOp>(op))
    return success();

  Attribute targetEncoding;
  for (Value result : op->getResults()) {
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy || !resultTy.getEncoding())
      continue;
    Attribute encoding =
        gluon_layout::unwrapNoVerifyEncoding(resultTy.getEncoding());
    if (!targetEncoding) {
      targetEncoding = encoding;
      continue;
    }
    if (targetEncoding != encoding)
      return op->emitError()
             << "same-encoding operation has conflicting result layouts: "
             << targetEncoding << " vs " << encoding;
  }

  if (!targetEncoding)
    return success();
  if (!isa<ttg::DotOperandEncodingAttr>(targetEncoding))
    return success();

  for (OpOperand &operandUse : op->getOpOperands()) {
    auto operandTy = dyn_cast<RankedTensorType>(operandUse.get().getType());
    if (!operandTy || !operandTy.getEncoding())
      continue;
    Attribute operandEncoding =
        gluon_layout::unwrapNoVerifyEncoding(operandTy.getEncoding());
    if (operandEncoding != targetEncoding)
      return op->emitError()
             << "dot-operand layout leaked into a same-encoding elementwise "
                "chain with mismatched operand layout "
             << operandEncoding << "; insert a boundary convert_layout before "
                "ordinary softmax/dS elementwise computation";
  }
  return success();
}

static LogicalResult verifyAsyncCopyContracts(
    ttg::AsyncCopyGlobalToLocalOp copyOp) {
  return c500::verifyC500AsyncCopyIssuePlan(copyOp);
}

static LogicalResult verifyDotMacaContract(tt::DotOp dotOp) {
  auto lhsType = dyn_cast<RankedTensorType>(dotOp.getA().getType());
  auto rhsType = dyn_cast<RankedTensorType>(dotOp.getB().getType());
  auto accumulatorType = dyn_cast<RankedTensorType>(dotOp.getC().getType());
  auto mma = accumulatorType
                 ? dyn_cast_or_null<ttg::MACAMmaEncodingAttr>(
                       accumulatorType.getEncoding())
                 : ttg::MACAMmaEncodingAttr();
  if (!lhsType || !rhsType || !mma)
    return success();
  if (gluon_layout::supportsC500MacaAccumulatorOrder(
          lhsType.getElementType(), rhsType.getElementType(),
          mma.getColMajor()))
    return success();
  return dotOp.emitError()
         << "C500 MMA accumulator order colMajor=" << mma.getColMajor()
         << " is not supported for operand element types "
         << lhsType.getElementType() << " and " << rhsType.getElementType();
}

static LogicalResult verifyLayoutContractsImpl(ModuleOp module) {
  if (failed(gluon_layout::verifyNoResidualCalls(
          module, "MetaX Gluon final layout verification")))
    return failure();
  if (failed(gluon_layout::verifyNoResidualNoVerifyEncodings(
          module, "MetaX Gluon final layout verification")))
    return failure();
  if (failed(gluon_layout::verifyNoResidualAutoEncodings(
          module, "MetaX Gluon final layout verification")))
    return failure();
  if (failed(gluon_layout::verifyMacaEncodingContracts(
          module, "MetaX Gluon final layout verification")))
    return failure();
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    LogicalResult status = TypeSwitch<Operation *, LogicalResult>(op)
                               .Case<ttg::LocalLoadOp>(verifyLocalLoadContract)
                               .Case<ttg::LocalStoreOp>(verifyLocalStoreContract)
                               .Case<ttg::LocalAllocOp>(
                                   verifyLocalAllocInitializerContract)
                               .Case<ttg::ExtractTensorOp>(
                                   gluon_layout::verifyExtractTensorContract)
                               .Case<ttg::InsertTensorOp>(
                                   gluon_layout::verifyInsertTensorContract)
                               .Case<ttg::AsyncCopyGlobalToLocalOp>(
                                   verifyAsyncCopyContracts)
                               .Case<ttg::BsmPermOp>(
                                   gluon_layout::verifyBsmPermPhysicalContract)
                               .Case<tt::DotOp>(verifyDotMacaContract)
                               .Case<tt::AtomicRMWOp>(
                                   gluon_layout::verifyAtomicRmwOwnership)
                               .Default(verifySameEncodingOp);
    return failed(status) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

class TritonMETAXGPUGluonVerifyLayoutContractsPass
    : public impl::TritonMETAXGPUGluonVerifyLayoutContractsBase<
          TritonMETAXGPUGluonVerifyLayoutContractsPass> {
public:
  void runOnOperation() override {
    if (failed(verifyLayoutContractsImpl(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult verifyTritonMETAXGPUGluonLayoutContracts(ModuleOp module) {
  return verifyLayoutContractsImpl(module);
}

std::unique_ptr<Pass>
createTritonMETAXGPUGluonVerifyLayoutContractsPass() {
  return std::make_unique<TritonMETAXGPUGluonVerifyLayoutContractsPass>();
}

} // namespace mlir
