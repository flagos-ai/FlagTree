
#include "magic-kernel/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <memory>

#define DEBUG_TYPE "materialize-strided-linalg-inputs"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MATERIALIZESTRIDEDLINALGINPUTS
#include "magic-kernel/Transforms/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static bool hasNonContiguousStrides(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return true;

  int64_t expected = 1;
  ArrayRef<int64_t> shape = type.getShape();

  for (int64_t i = type.getRank() - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(shape[i]) || ShapedType::isDynamic(strides[i]))
      return true;

    if (shape[i] == 1)
      continue;

    if (strides[i] != expected)
      return true;

    expected *= shape[i];
  }

  return false;
}

static bool isComputeGeneric(linalg::GenericOp op) {
  Block &body = op.getRegion().front();

  Operation *computeOp = nullptr;
  for (Operation &inner : body.without_terminator()) {
    if (computeOp)
      return false; // more than one non-yield op

    computeOp = &inner;
  }

  if (!computeOp)
    return false;

  return computeOp->getDialect()->getNamespace() ==
         arith::ArithDialect::getDialectNamespace();
}

struct MaterializeStridedLinalgInputsPass
    : public triton::impl::MaterializeStridedLinalgInputsBase<
          MaterializeStridedLinalgInputsPass> {
  using MaterializeStridedLinalgInputsBase<
      MaterializeStridedLinalgInputsPass>::MaterializeStridedLinalgInputsBase;

  void process(func::FuncOp &func) {
    IRRewriter rewriter(func.getContext());

    SmallVector<linalg::GenericOp> generics;
    func.walk([&](linalg::GenericOp op) { generics.push_back(op); });

    for (linalg::GenericOp generic : generics) {
      if (!isComputeGeneric(generic))
        continue;

      rewriter.setInsertionPoint(generic);
      Location loc = generic.getLoc();

      for (OpOperand *inputOperand : generic.getDpsInputOperands()) {
        Value input = inputOperand->get();
        auto subview = input.getDefiningOp<memref::SubViewOp>();
        if (!subview)
          continue;

        auto subviewType = dyn_cast<MemRefType>(subview.getType());
        if (!subviewType || !hasNonContiguousStrides(subviewType))
          continue;

        SmallVector<Value> dynamicSizes;
        for (auto [idx, dim] : llvm::enumerate(subviewType.getShape())) {
          if (ShapedType::isDynamic(dim)) {
            dynamicSizes.push_back(
                rewriter.create<memref::DimOp>(loc, subview, idx));
          }
        }

        auto allocType = MemRefType::get(
            subviewType.getShape(), subviewType.getElementType(),
            MemRefLayoutAttrInterface{}, subviewType.getMemorySpace());

        Value alloc =
            rewriter.create<memref::AllocOp>(loc, allocType, dynamicSizes);

        rewriter.create<memref::CopyOp>(loc, subview, alloc);

        generic->setOperand(inputOperand->getOperandNumber(), alloc);
      }
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod->walk<WalkOrder::PreOrder>([&](func::FuncOp func) { process(func); });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createMaterializeStridedLinalgInputsPass() {
  return std::make_unique<MaterializeStridedLinalgInputsPass>();
}