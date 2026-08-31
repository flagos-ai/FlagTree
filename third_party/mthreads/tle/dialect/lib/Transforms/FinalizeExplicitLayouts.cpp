#ifdef __TLE__

#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLEFINALIZEEXPLICITLAYOUTS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

static LogicalResult verifyExplicitResultEncoding(Operation *op,
                                                  NamedAttribute namedAttr,
                                                  StringRef prefix) {
  StringRef name = namedAttr.getName().getValue();
  StringRef suffix = name.drop_front(prefix.size());
  unsigned resultNumber = 0;
  if (suffix.empty() || suffix.getAsInteger(10, resultNumber))
    return op->emitOpError("has malformed explicit MUSA TLE result encoding '")
           << name << "'";
  if (resultNumber >= op->getNumResults())
    return op->emitOpError("has explicit MUSA TLE encoding for missing result ")
           << resultNumber;

  auto resultType =
      dyn_cast<RankedTensorType>(op->getResult(resultNumber).getType());
  if (!resultType || !resultType.getEncoding())
    return op->emitOpError("has explicit MUSA TLE encoding on result ")
           << resultNumber << " without a distributed ranked tensor type";
  if (resultType.getEncoding() != namedAttr.getValue())
    return op->emitOpError("has explicit MUSA TLE result encoding that does "
                           "not match the final tensor type encoding:\n  ")
           << namedAttr.getValue() << "\nand\n  " << resultType.getEncoding();
  return success();
}

static LogicalResult verifyExplicitMemoryEncoding(Operation *op,
                                                  Attribute encoding) {
  Value pointer = getMemAccessPtr(op);
  if (!pointer)
    return op->emitOpError(
        "has explicit MUSA TLE memory encoding on a non-memory operation");

  auto pointerType = dyn_cast<RankedTensorType>(pointer.getType());
  if (!pointerType || !pointerType.getEncoding())
    return op->emitOpError(
        "has explicit MUSA TLE memory encoding without an encoded tensor "
        "pointer");
  if (pointerType.getEncoding() != encoding)
    return op->emitOpError("has explicit MUSA TLE memory encoding that does "
                           "not match the final pointer encoding:\n  ")
           << encoding << "\nand\n  " << pointerType.getEncoding();

  Attribute inferredEncoding;
  if (failed(inferTleExplicitMemoryEncoding(op, inferredEncoding)))
    return failure();
  if (inferredEncoding != encoding)
    return op->emitOpError(
        "has an inconsistent inferred explicit MUSA TLE memory encoding");
  return success();
}

class FinalizeExplicitLayoutsPass
    : public impl::TritonMUSAGPUTLEFinalizeExplicitLayoutsBase<
          FinalizeExplicitLayoutsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    StringRef resultPrefix = getTleExplicitEncodingAttrPrefix();
    StringRef memoryName = getTleExplicitMemoryEncodingAttrName();

    WalkResult verification = module.walk([&](Operation *op) -> WalkResult {
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name.starts_with(resultPrefix) &&
            failed(verifyExplicitResultEncoding(op, attr, resultPrefix)))
          return WalkResult::interrupt();
      }
      if (Attribute memoryEncoding = op->getAttr(memoryName)) {
        if (failed(verifyExplicitMemoryEncoding(op, memoryEncoding)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (verification.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    SmallVector<triton::gpu::ConvertLayoutOp> identityConversions;
    module.walk([&](Operation *op) {
      SmallVector<StringAttr> attrsToRemove;
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name.starts_with(resultPrefix) || name == memoryName)
          attrsToRemove.push_back(attr.getName());
      }
      for (StringAttr name : attrsToRemove)
        op->removeAttr(name);

      if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        if (convert.getSrc().getType() == convert.getType())
          identityConversions.push_back(convert);
      }
    });

    for (triton::gpu::ConvertLayoutOp convert : identityConversions) {
      convert.getResult().replaceAllUsesWith(convert.getSrc());
      convert.erase();
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
