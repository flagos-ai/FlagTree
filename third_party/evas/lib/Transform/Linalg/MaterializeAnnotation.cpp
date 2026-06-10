#include "epu/memory.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstdint>

#define GEN_PASS_DEF_MATERIALIZEANNOTATION
#include "evas/Transform/Linalg/Passes.h.inc"
namespace mlir::triton::ev {

namespace {
/// A pass to insert deallocations for allocated buffers after theirlast use.
using namespace mlir;

// Modify memory scope for connected memref values recursively
void InferMemrefType(Value value, bool onlyScope = true) {
  // Get the type of input value
  Type inputType = value.getType();

  // Get all users of this value
  for (Operation *user : value.getUsers()) {
    // Only infer op with ViewLikeOpInterface
    if (!dyn_cast<mlir::ViewLikeOpInterface>(user))
      continue;

    // For each result of the user operation
    for (Value result : user->getResults()) {
      // If result type can be cast to MemRefType
      if (MemRefType resultMemRef = dyn_cast<MemRefType>(result.getType())) {
        // Create new memref type with same properties but input memory scope

        if (MemRefType inputMemRef = dyn_cast<MemRefType>(inputType)) {
          if (onlyScope) {
            auto newType = MemRefType::get(
                resultMemRef.getShape(), resultMemRef.getElementType(),
                resultMemRef.getLayout(),
                IntegerAttr::get(IntegerType::get(value.getContext(), 64),
                                 (int64_t)inputMemRef.getMemorySpaceAsInt()));
            // Update the result type
            result.setType(newType);
          } else {
            result.setType(inputMemRef);
          }
          // Recursively update memory scope for connected values
          InferMemrefType(result, onlyScope);
        }
      }
    }
  }
}

void foldMemrefCopy(ModuleOp moduleOp) {
  moduleOp.walk([&](memref::CopyOp copyOp) {
    Value src = copyOp.getSource();
    Value dst = copyOp.getTarget();

    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto dstType = dyn_cast<MemRefType>(dst.getType());

    if (!srcType || !dstType || dstType != srcType)
      return;
    // Replace all uses of dst with src and erase the copy op
    dst.replaceAllUsesWith(src);
    copyOp.erase();
  });
}

struct MaterializeAnnotationPass
    : public ::impl::MaterializeAnnotationBase<MaterializeAnnotationPass> {

  void materializeAddress(tts::AnnotateOp annotateOp, OpBuilder &builder) {
    Value input = annotateOp.getSrc();
    auto allocOp = input.getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return;

    auto memInfoAttr = annotateOp.getMeminfoAttr();
    if (!memInfoAttr)
      return;

    auto memrefType = cast<MemRefType>(allocOp.getType());
    int memScope = (int)memInfoAttr.getScope();
    auto address = memInfoAttr.getAddress();

    // memscope should be handled already
    assert(memScope == memrefType.getMemorySpaceAsInt());

    if (address > 0) {
      allocOp->setDiscardableAttr(mlir::ev::phyAddrName,
                                  builder.getI64IntegerAttr(address));
    }

  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    // Walk through all AnnotateOp operations in the module
    moduleOp.walk([&](tts::AnnotateOp annotateOp) {
      materializeAddress(annotateOp, builder);
    });

    // Remove all annotate ops
    moduleOp.walk([&](tts::AnnotateOp annotateOp) { annotateOp.erase(); });

  }
};

} // namespace
std::unique_ptr<Pass> createMaterializeAnnotationPass() {
  return std::make_unique<MaterializeAnnotationPass>();
}
} // namespace mlir::triton::ev
